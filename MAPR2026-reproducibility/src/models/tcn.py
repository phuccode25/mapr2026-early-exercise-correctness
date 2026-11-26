import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    """
    Basic temporal convolution block.

    Input shape:
        (B, C, T)

    Output shape:
        (B, hidden, T)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual

        out = self.relu2(out)
        out = self.drop2(out)

        return out


class MultiTaskTCN(nn.Module):
    """
    Multi-task TCN for REHAB24-6.

    Input:
        x: (B, T, F)
        Example: (B, 64, 78)

    Outputs:
        action_logits: (B, 6)
        correctness_logits: (B,)
    """

    def __init__(self, input_dim=78, hidden_dim=128, num_actions=6, dropout=0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            TCNBlock(input_dim, hidden_dim, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim, hidden_dim, dilation=2, dropout=dropout),
            TCNBlock(hidden_dim, hidden_dim, dilation=4, dropout=dropout),
            TCNBlock(hidden_dim, hidden_dim, dilation=8, dropout=dropout),
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )

        self.correctness_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, return_embedding=False):
        # x: (B, T, F)
        # Conv1d needs: (B, F, T)
        x = x.transpose(1, 2)

        h = self.encoder(x)  # (B, hidden, T)

        # Simple global average pooling over time.
        embedding = h.mean(dim=2)  # (B, hidden)

        action_logits = self.action_head(embedding)
        correctness_logits = self.correctness_head(embedding).squeeze(1)

        if return_embedding:
            return action_logits, correctness_logits, embedding

        return action_logits, correctness_logits