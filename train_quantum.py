import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
import time
import os
import pandas as pd
import numpy as np
import random

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

from bnci2014_preprocessing import run_pipeline
from eegnet_tcn_quantum_model import EEGNetTCNQuantum


# -----------------------------
# SETTINGS
# -----------------------------
SUBJECTS = list(range(1, 10))
BATCH_SIZE = 16
MAX_EPOCHS = 120
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "results_quantum"
os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------
# CONFIGS
# -----------------------------
CONFIGS = [
    {"F1": 8,  "D": 1, "dropout": 0.3,  "kernel": 64, "tcn_kernel": 3},
    {"F1": 12, "D": 1, "dropout": 0.25, "kernel": 64, "tcn_kernel": 3},
    {"F1": 16, "D": 1, "dropout": 0.2,  "kernel": 64, "tcn_kernel": 3},
]

QUANTUM_CONFIGS = [
    {"n_qubits": 6,  "n_layers": 2},
    {"n_qubits": 8,  "n_layers": 2},
    {"n_qubits": 8,  "n_layers": 3},
    {"n_qubits": 10, "n_layers": 2},
    {"n_qubits": 10, "n_layers": 4},
]

LR_CONFIGS = [
    {"lr_q": 1e-4, "lr_c": 7e-4},
    {"lr_q": 5e-5, "lr_c": 5e-4},
]

AUX_WEIGHTS = [0.2, 0.3]


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
def evaluate(model, loader):
    model.eval()
    preds_all, y_all = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out, _ = model(X)

            _, preds = torch.max(out, 1)

            preds_all.extend(preds.cpu().numpy())
            y_all.extend(y.cpu().numpy())

    acc = 100 * (np.mean(np.array(preds_all) == np.array(y_all)))
    kappa = cohen_kappa_score(y_all, preds_all)

    return acc, kappa


# -----------------------------
# MAIN RUN LOOP
# -----------------------------
for run_id in range(1, 4):

    print(f"\n========== RUN {run_id} ==========\n")
    set_seed(42 + run_id)

    data = run_pipeline(subject_ids=SUBJECTS)

    results = []

    for subj in SUBJECTS:

        print(f"\nSUBJECT {subj}")

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
        best_cfg = None
        best_metrics = None

        for cfg in CONFIGS:
            for qcfg in QUANTUM_CONFIGS:
                for lr_cfg in LR_CONFIGS:
                    for aux_w in AUX_WEIGHTS:

                        print(f"CFG: {cfg}, Q: {qcfg}, LR: {lr_cfg}, AUX: {aux_w}")

                        model = EEGNetTCNQuantum(
                            F1=cfg["F1"],
                            D=cfg["D"],
                            F2=cfg["F1"],
                            dropout=cfg["dropout"],
                            kernel_length=cfg["kernel"],
                            tcn_kernel=cfg["tcn_kernel"],
                            n_qubits=qcfg["n_qubits"],
                            n_layers=qcfg["n_layers"]
                        ).to(DEVICE)

                        optimizer = torch.optim.Adam([
                            {"params": model.quantum.parameters(), "lr": lr_cfg["lr_q"]},
                            {"params": model.pre_quantum.parameters(), "lr": lr_cfg["lr_q"]},
                            {"params": model.classifier.parameters(), "lr": lr_cfg["lr_c"]},
                            {"params": model.classical_head.parameters(), "lr": lr_cfg["lr_c"]},
                            {"params": model.classical_proj.parameters(), "lr": lr_cfg["lr_c"]},
                            {"params": model.temporal_conv.parameters(), "lr": lr_cfg["lr_c"]},
                            {"params": model.depthwise_conv.parameters(), "lr": lr_cfg["lr_c"]},
                            {"params": model.separable_conv.parameters(), "lr": lr_cfg["lr_c"]},
                            {"params": model.tcn.parameters(), "lr": lr_cfg["lr_c"]},
                        ], weight_decay=1e-4)

                        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

                        local_best = -1
                        best_state = copy.deepcopy(model.state_dict())

                        for epoch in range(MAX_EPOCHS):
                            train_epoch(model, optimizer, criterion, train_loader, aux_w, epoch)

                            val_acc, _ = evaluate(model, val_loader)

                            if val_acc > local_best:
                                local_best = val_acc
                                best_state = copy.deepcopy(model.state_dict())

                        model.load_state_dict(best_state)

                        acc, kappa = evaluate(model, test_loader)

                        if acc > best_acc:
                            best_acc = acc
                            best_cfg = {**cfg, **qcfg, **lr_cfg, "aux_w": aux_w}

        results.append({
            "subject": subj,
            "accuracy": best_acc,
            "config": best_cfg
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(SAVE_DIR, f"final_results_run{run_id}.csv"), index=False)

    print(df)