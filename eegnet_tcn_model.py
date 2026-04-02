import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# ─────────────────────────────
# TCN BLOCK
# ─────────────────────────────
class TCNResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad, dilation=dilation, bias=False
        ))
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad, dilation=dilation, bias=False
        ))
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        T = x.size(2)

        out = self.conv1(x)[:, :, :T]
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)[:, :, :T]
        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop2(out)

        return out + x


class TCN(nn.Module):
    def __init__(self, channels, dilations=(1,2,4,8,16), dropout=0.2, kernel_size=3):
        super().__init__()
        self.blocks = nn.Sequential(*[
            TCNResidualBlock(channels, kernel_size, d, dropout)
            for d in dilations
        ])

    def forward(self, x):
        return self.blocks(x)


# ─────────────────────────────
# EEGNet + TCN
# ─────────────────────────────
class EEGNetTCN(nn.Module):
    def __init__(self, n_channels=22, n_classes=4,
                 F1=16, D=2, F2=32, dropout=0.25,
                 kernel_length=64, sep_kernel=16, tcn_kernel=3):
        super().__init__()

        # 🔥 CHANNEL DROPOUT
        self.channel_dropout = nn.Dropout2d(p=0.1)

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

        self.tcn = TCN(F2, dropout=dropout, kernel_size=tcn_kernel)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(F2, n_classes)

    def forward(self, x):
        x = self.channel_dropout(x)

        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)

        x = x.squeeze(2)
        x = self.tcn(x)

        x = self.gap(x).squeeze(-1)
        return self.fc(x)