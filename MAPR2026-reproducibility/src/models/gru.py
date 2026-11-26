import torch
import torch.nn as nn


class SequenceRNNEncoder(nn.Module):
    """
    Shared GRU/LSTM encoder for REHAB24 sequence baselines.

    Input:
        x: (B, T, F), e.g. (B, 64, 78)
    """

    def __init__(
        self,
        input_dim=78,
        hidden_dim=128,
        num_layers=1,
        dropout=0.1,
        rnn_type="gru",
        bidirectional=False,
    ):
        super().__init__()

        rnn_type = rnn_type.lower()
        if rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        recurrent_dropout = dropout if num_layers > 1 else 0.0

        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        # Mean pooling keeps the baseline comparable to the TCN global pooling.
        return self.dropout(out.mean(dim=1))


class MultiTaskRNN(nn.Module):
    """
    Multi-task GRU/LSTM baseline.

    Outputs:
        action_logits: (B, num_actions)
        correctness_logits: (B,)
    """

    def __init__(
        self,
        input_dim=78,
        hidden_dim=128,
        num_actions=6,
        num_layers=1,
        dropout=0.1,
        rnn_type="gru",
        bidirectional=False,
    ):
        super().__init__()

        self.encoder = SequenceRNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
        )

        self.action_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )
        self.correctness_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        action_logits = self.action_head(embedding)
        correctness_logits = self.correctness_head(embedding).squeeze(1)
        return action_logits, correctness_logits


class CorrectnessOnlyRNN(nn.Module):
    """
    Correctness-only GRU/LSTM baseline.
    """

    def __init__(
        self,
        input_dim=78,
        hidden_dim=128,
        num_layers=1,
        dropout=0.1,
        rnn_type="gru",
        bidirectional=False,
    ):
        super().__init__()

        self.encoder = SequenceRNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
        )

        self.correctness_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return self.correctness_head(embedding).squeeze(1)
