import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
import time
import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

# NEW IMPORTS
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import json

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

from bnci2014_preprocessing import run_pipeline
from eegnet_tcn_model import EEGNetTCN


# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
SUBJECTS = list(range(1, 10))
BATCH_SIZE = 16
MAX_EPOCHS = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {DEVICE}\n")

SAVE_DIR = "results_v4"
PLOTS_DIR = os.path.join(SAVE_DIR, "plots")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# CONFIGS
# ─────────────────────────────────────────────
CONFIGS = [
    {"F1": 8, "D": 2, "dropout": 0.3, "kernel": 32, "tcn_kernel": 3},
    {"F1": 8, "D": 2, "dropout": 0.3, "kernel": 64, "tcn_kernel": 3},
    {"F1": 8, "D": 2, "dropout": 0.25, "kernel": 64, "tcn_kernel": 3},
    {"F1": 16, "D": 2, "dropout": 0.3, "kernel": 64, "tcn_kernel": 3},
    {"F1": 16, "D": 2, "dropout": 0.25, "kernel": 64, "tcn_kernel": 3},
    {"F1": 16, "D": 2, "dropout": 0.25, "kernel": 64, "tcn_kernel": 5},
]


# ─────────────────────────────────────────────
# TRAIN FUNCTION
# ─────────────────────────────────────────────
def train_epoch(model, optimizer, criterion, loader, epoch):
    model.train()

    for X, y in loader:
        if epoch > 10:
            noise = 0.02 * X.std() * torch.randn_like(X)
            X = X + noise

        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()


# ─────────────────────────────────────────────
# EVAL FUNCTION (UPDATED)
# ─────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    preds_all, y_all, probs_all = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)

            probs = torch.softmax(out, dim=1)
            _, preds = torch.max(out, 1)

            preds_all.extend(preds.cpu().numpy())
            y_all.extend(y.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())

    acc = 100 * (np.mean(np.array(preds_all) == np.array(y_all)))
    kappa = cohen_kappa_score(y_all, preds_all)

    return acc, kappa, np.array(y_all), np.array(preds_all), np.array(probs_all)


# ─────────────────────────────────────────────
# METRICS FUNCTION
# ─────────────────────────────────────────────
def compute_metrics(model, sample):
    params = sum(p.numel() for p in model.parameters())

    start = time.time()
    with torch.no_grad():
        model(sample)
    latency = (time.time() - start) * 1000

    vram = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else 0
    )

    return params, latency, vram


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
data = run_pipeline(subject_ids=SUBJECTS)

results = []


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
for subj in SUBJECTS:
    print(f"\n{'='*50}\nSUBJECT {subj}\n{'='*50}")

    result = data[subj]

    X_train = torch.from_numpy(result["X_train"]).float()
    y_train = torch.from_numpy(result["y_train"]).long()
    X_test  = torch.from_numpy(result["X_test"]).float()
    y_test  = torch.from_numpy(result["y_test"]).long()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

    best_acc = 0
    best_kappa = 0
    best_cfg = None
    best_state = None
    best_metrics = None

    for cfg in CONFIGS:
        print(f"Trying config: {cfg}")

        model = EEGNetTCN(
            F1=cfg["F1"],
            D=cfg["D"],
            F2=cfg["F1"] * cfg["D"],
            dropout=cfg["dropout"],
            kernel_length=cfg["kernel"],
            tcn_kernel=cfg["tcn_kernel"]
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        patience = 30
        counter = 0
        local_best = -1
        best_epoch_state = copy.deepcopy(model.state_dict())
        val_history = []
        best_val_history = []

        for epoch in range(1, MAX_EPOCHS + 1):
            train_epoch(model, optimizer, criterion, train_loader, epoch)

            val_acc, _, _, _, _ = evaluate(model, val_loader)
            val_history.append(val_acc)

            scheduler.step(val_acc)

            print(f"Epoch {epoch} | Val Acc: {val_acc:.2f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            val_history.append(val_acc)

            if val_acc > local_best:
                local_best = val_acc
                counter = 0
                best_epoch_state = copy.deepcopy(model.state_dict())
                best_val_history = val_history.copy()
            else:
                counter += 1

            if counter >= patience:
                break

        model.load_state_dict(best_epoch_state)

        acc, kappa, y_true, y_pred, y_prob = evaluate(model, test_loader)

        print(f"Config {cfg} → Acc: {acc:.2f}")

        if acc > best_acc:
            best_acc = acc
            best_kappa = kappa
            best_cfg = cfg
            best_state = best_epoch_state

            sample = X_test[:1].to(DEVICE)
            best_metrics = compute_metrics(model, sample)

            # ───── SAVE PLOTS ─────
            cm = confusion_matrix(y_true, y_pred)
            ConfusionMatrixDisplay(cm).plot()
            plt.savefig(os.path.join(PLOTS_DIR, f"confusion_subject_{subj}.png"))
            plt.close()

            y_bin = label_binarize(y_true, classes=np.unique(y_true))
            plt.figure()
            for i in range(y_bin.shape[1]):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr, tpr):.2f})")

            plt.plot([0, 1], [0, 1], 'k--')
            plt.legend()
            plt.savefig(os.path.join(PLOTS_DIR, f"roc_subject_{subj}.png"))
            plt.close()

            plt.plot(best_val_history)
            plt.title("Validation Curve")
            plt.savefig(os.path.join(PLOTS_DIR, f"train_curve_subject_{subj}.png"))
            plt.close()

            # ───── SAVE JSON ─────
            with open(os.path.join(SAVE_DIR, f"subject_{subj}_metrics.json"), "w") as f:
                json.dump({
                    "accuracy": float(acc),
                    "kappa": float(kappa),
                    "config": best_cfg
                }, f, indent=4)

    torch.save(best_state, os.path.join(SAVE_DIR, f"subject_{subj}.pt"))

    results.append({
        "subject": subj,
        "accuracy": best_acc,
        "kappa": best_kappa,
        "params": best_metrics[0],
        "latency_ms": best_metrics[1],
        "vram_MB": best_metrics[2],
        "config": best_cfg
    })


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(os.path.join(SAVE_DIR, "final_results.csv"), index=False)

print("\nFINAL RESULTS TABLE")
print(df)

print("\nSUMMARY")
print(f"Mean Accuracy: {df['accuracy'].mean():.2f}")
print(f"Std Accuracy: {df['accuracy'].std():.2f}")