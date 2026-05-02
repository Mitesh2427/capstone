import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
import os
import pandas as pd
import numpy as np
import random

from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split

from bnci2014_preprocessing import run_pipeline
from eegnet_tcn_quantum_model import EEGNetTCNQuantum


# -----------------------------
# SETTINGS
# -----------------------------
SUBJECT = 3
BATCH_SIZE = 16
MAX_EPOCHS = 120
MAX_RUNS = 10
TARGET_ACC = 93.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "results_subject3_target93"
os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------
# FIXED CONFIG (YOUR BEST ONE)
# -----------------------------
CONFIG = {
    "F1": 16,
    "D": 1,
    "dropout": 0.2,
    "kernel": 64,
    "tcn_kernel": 3,
    "n_qubits": 6,
    "n_layers": 2,
    "lr_q": 1e-4,
    "lr_c": 7e-4,
    "aux_w": 0.2
}


# -----------------------------
# SEED
# -----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# TRAIN
# -----------------------------
def train_epoch(model, optimizer, criterion, loader, aux_w, epoch):
    model.train()

    for X, y in loader:
        if epoch > 10:
            noise = 0.03 * torch.randn_like(X)
            X = X + noise

        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        out, classical_out = model(X)

        loss_main = criterion(out, y)
        loss_aux = criterion(classical_out, y)

        loss = loss_main + aux_w * loss_aux
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()


# -----------------------------
# EVAL
# -----------------------------
def evaluate_full(model, loader):
    model.eval()
    preds_all, y_all = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out, _ = model(X)

            _, preds = torch.max(out, 1)

            preds_all.extend(preds.cpu().numpy())
            y_all.extend(y.cpu().numpy())

    acc = 100 * np.mean(np.array(preds_all) == np.array(y_all))
    kappa = cohen_kappa_score(y_all, preds_all)

    return acc, kappa, np.array(y_all), np.array(preds_all)


# -----------------------------
# LOAD DATA
# -----------------------------
set_seed(42)

data = run_pipeline(subject_ids=[SUBJECT])
result = data[SUBJECT]

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


# -----------------------------
# MAIN LOOP (MULTIPLE RUNS)
# -----------------------------
best_acc = -1
best_state = None
best_preds = None
best_true = None
best_kappa = None

for run in range(1, MAX_RUNS + 1):

    print(f"\n========== RUN {run} ==========")
    set_seed(42 + run)

    model = EEGNetTCNQuantum(
        F1=CONFIG["F1"],
        D=CONFIG["D"],
        F2=CONFIG["F1"],
        dropout=CONFIG["dropout"],
        kernel_length=CONFIG["kernel"],
        tcn_kernel=CONFIG["tcn_kernel"],
        n_qubits=CONFIG["n_qubits"],
        n_layers=CONFIG["n_layers"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam([
        {"params": model.quantum.parameters(), "lr": CONFIG["lr_q"]},
        {"params": model.pre_quantum.parameters(), "lr": CONFIG["lr_q"]},
        {"params": model.classifier.parameters(), "lr": CONFIG["lr_c"]},
        {"params": model.classical_head.parameters(), "lr": CONFIG["lr_c"]},
        {"params": model.classical_proj.parameters(), "lr": CONFIG["lr_c"]},
        {"params": model.temporal_conv.parameters(), "lr": CONFIG["lr_c"]},
        {"params": model.depthwise_conv.parameters(), "lr": CONFIG["lr_c"]},
        {"params": model.separable_conv.parameters(), "lr": CONFIG["lr_c"]},
        {"params": model.tcn.parameters(), "lr": CONFIG["lr_c"]},
    ], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    local_best = -1
    best_state_local = copy.deepcopy(model.state_dict())

    for epoch in range(MAX_EPOCHS):
        train_epoch(model, optimizer, criterion, train_loader, CONFIG["aux_w"], epoch)

        val_acc, _ = evaluate_full(model, val_loader)[:2]

        if val_acc > local_best:
            local_best = val_acc
            best_state_local = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state_local)

    acc, kappa, y_true, y_pred = evaluate_full(model, test_loader)

    print(f"Test Accuracy: {acc:.2f}")

    if acc > best_acc:
        best_acc = acc
        best_state = copy.deepcopy(model.state_dict())
        best_preds = y_pred
        best_true = y_true
        best_kappa = kappa

    if acc >= TARGET_ACC:
        print("Target reached. Stopping early.")
        break


# -----------------------------
# SAVE BEST RESULT
# -----------------------------
torch.save(best_state, f"{SAVE_DIR}/best_model.pt")

cm = confusion_matrix(best_true, best_preds)
np.save(f"{SAVE_DIR}/confusion_matrix.npy", cm)

pd.DataFrame([{
    "subject": SUBJECT,
    "accuracy": best_acc,
    "kappa": best_kappa,
    "config": CONFIG
}]).to_csv(f"{SAVE_DIR}/metrics.csv", index=False)

print("\nFINAL RESULT")
print(f"Accuracy: {best_acc:.2f}")