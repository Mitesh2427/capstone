from eegnet_tcn_model import TCN
from quantum_layer import QuantumLayer
import torch
import torch.nn as nn


class EEGNetTCNQuantum(nn.Module):
    def __init__(self, n_channels=22, n_classes=4,
                 F1=12, D=1, F2=12, dropout=0.25,
                 kernel_length=64, sep_kernel=16, tcn_kernel=3,
                 n_qubits=10, n_layers=4):

        super().__init__()

        # -----------------------------
        # EEGNet
        # -----------------------------
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU()
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, sep_kernel),
                      padding=(0, sep_kernel // 2),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # -----------------------------
        # TCN
        # -----------------------------
        self.tcn = TCN(F2, dropout=dropout, kernel_size=tcn_kernel)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # -----------------------------
        # Quantum
        # -----------------------------
        self.n_qubits = n_qubits

        # Improved projection (less info loss)
        self.pre_quantum = nn.Sequential(
            nn.Linear(F2, 2 * n_qubits),
            nn.ReLU(),
            nn.Linear(2 * n_qubits, n_qubits),
            nn.LayerNorm(n_qubits)
        )

        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)

        # Classical projection (for fusion)
        self.classical_proj = nn.Linear(F2, n_qubits)

        # Gate
        self.gate_layer = nn.Linear(F2, n_qubits)

        # Classical head (aux loss + confidence)
        self.classical_head = nn.Linear(F2, n_classes)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        # -----------------------------
        # EEGNet + TCN
        # -----------------------------
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)

        x = x.squeeze(2)
        x = self.tcn(x)

        x = self.gap(x).squeeze(-1)

        x_classical = x

        # -----------------------------
        # Classical branch
        # -----------------------------
        classical_logits = self.classical_head(x_classical)

        probs = torch.softmax(classical_logits, dim=1)
        confidence = torch.max(probs, dim=1, keepdim=True)[0]

        # -----------------------------
        # Quantum branch
        # -----------------------------
        x_q = self.pre_quantum(x_classical)

        # Noise regularization (important)
        x_q = x_q + 0.01 * torch.randn_like(x_q)

        x_q = self.quantum(x_q)

        # -----------------------------
        # Fusion
        # -----------------------------
        x_classical_proj = self.classical_proj(x_classical)

        gate = torch.sigmoid(self.gate_layer(x_classical))

        # Confidence-aware gating
        confidence = confidence.repeat(1, self.n_qubits)
        gate = gate * confidence

        # Stabilize gate
        gate = torch.clamp(gate, 0.2, 0.8)

        # Combine
        x = gate * x_q + (1 - gate) * x_classical_proj

        # Residual safety
        x = x + 0.3 * x_classical_proj

        # -----------------------------
        # Final output
        # -----------------------------
        final_logits = self.classifier(x)

        return final_logits, classical_logits