import torch
import torch.nn as nn

from src.models.tcn import TCNBlock


class CorrectnessOnlyTCN(nn.Module):
    """
    TCN model for correctness classification only.

    Input:
        x: (B, T, F), e.g. (B, 64, 78)

    Output:
        correctness_logits: (B,)
    """

    def __init__(self, input_dim=78, hidden_dim=128, dropout=0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            TCNBlock(input_dim, hidden_dim, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim, hidden_dim, dilation=2, dropout=dropout),
            TCNBlock(hidden_dim, hidden_dim, dilation=4, dropout=dropout),
            TCNBlock(hidden_dim, hidden_dim, dilation=8, dropout=dropout),
        )

        self.correctness_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (B, T, F)
        # Conv1d expects: (B, F, T)
        x = x.transpose(1, 2)

        h = self.encoder(x)  # (B, hidden, T)

        embedding = h.mean(dim=2)  # (B, hidden)

        correctness_logits = self.correctness_head(embedding).squeeze(1)

        return correctness_logits