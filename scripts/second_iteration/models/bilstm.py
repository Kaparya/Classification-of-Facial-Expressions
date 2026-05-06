import torch
import torch.nn as nn

from config import IEMOCAP_CKPT_DIR, FEAT_EFF, N_CLS, DEVICE


class BiLSTMClassifier(nn.Module):
    def __init__(self, feat_dim: int = 1280, hidden: int = 128,
                 n_classes: int = 6, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            feat_dim, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.bilstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(h)


def load_bilstm() -> BiLSTMClassifier:
    model = BiLSTMClassifier(FEAT_EFF, hidden=128, n_classes=N_CLS).to(DEVICE)
    state = torch.load(IEMOCAP_CKPT_DIR / "bilstm_ft.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
