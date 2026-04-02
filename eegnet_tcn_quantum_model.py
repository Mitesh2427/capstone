from eegnet_tcn_model import TCN
from quantum_layer import QuantumLayer
import torch
import torch.nn as nn


class EEGNetTCNQuantum(nn.Module):
    def __init__(self, n_channels=22, n_classes=4,
                 F1=8, D=2, F2=16, dropout=0.2,
                 kernel_length=64, sep_kernel=16, tcn_kernel=3):

        super().__init__()

        # ───── EEGNet BLOCK ─────
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length),
                      padding=(0, kernel_length // 2 - 1), bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU()
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D,
                      kernel_size=(1, sep_kernel),
                      padding=(0, sep_kernel // 2),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # ───── TCN ─────
        self.tcn = TCN(F2, dropout=dropout, kernel_size=tcn_kernel)

        self.pre_gap_dropout = nn.Dropout(0.2)

        # ───── HEAD ─────
        self.gap = nn.AdaptiveAvgPool1d(1)


        self.n_qubits = 4


        # 🔥 simple projection (NO deep MLP)
        self.pre_quantum = nn.Linear(F2, self.n_qubits)

        # 🔥 quantum layer (light)
        self.quantum = QuantumLayer(n_qubits=self.n_qubits, n_layers=2)

        # 🔥 NEW CLASSICAL CLASSIFIER (IMPORTANT)
        self.classifier = nn.Sequential(
            nn.Linear(F2 + self.n_qubits, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)

        x = x.squeeze(2)
        x = self.tcn(x)

        x = self.pre_gap_dropout(x)

        x = self.gap(x).squeeze(-1)

        x_orig = x

        x_q = self.pre_quantum(x)
        x_q = self.quantum(x_q)

        x = torch.cat([x_orig, x_q], dim=1)

        x = self.classifier(x)

        return x