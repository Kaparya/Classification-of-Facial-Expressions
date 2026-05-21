"""BiLSTM bimodal cascade head — second iteration.

Late-fusion of video (BiLSTM over EmotiEffNet features) and audio
(MelCNN gate's conv stack + 256-d projection, binary head dropped).
Trained per-LOSO-fold on IEMOCAP by `notebooks/iemocap_benchmark_4class.ipynb`
under the slug `ours_bilstm_bimodal_hard_skip` -- the final shipped variant
(pretrained-init, hard cascade on validation + inference). Audio input is the
SAME (1,1,64,128) log-mel tensor consumed by the cascade gate -- the head
reuses it without any re-extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BIMODAL_CKPT_DIR, BIMODAL_SESSION, FEAT_EFF, N_CLS, N_MELS, MAX_LEN, DEVICE


class BiLSTMBimodal(nn.Module):
    """Mirror of `BiLSTMBimodal` in iemocap_benchmark_4class.ipynb (cell 12).
    forward(video_eff, audio_mel) -> logits (B, N_CLS)."""

    def __init__(self, feat_dim_v: int = 1280, hidden: int = 128,
                 n_classes: int = 4, n_mels: int = 64, mel_len: int = 128,
                 audio_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.bilstm_v = nn.LSTM(feat_dim_v, hidden, num_layers=1,
                                batch_first=True, bidirectional=True)
        self.audio_features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        flat_dim = 64 * (n_mels // 8) * (mel_len // 8)
        self.audio_proj = nn.Linear(flat_dim, audio_dim)
        self.audio_drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden + audio_dim, 64), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(64, n_classes),
        )

    def forward(self, video: torch.Tensor, audio_mel: torch.Tensor) -> torch.Tensor:
        _, (hv, _) = self.bilstm_v(video)
        h_v = torch.cat([hv[-2], hv[-1]], dim=-1)
        x = self.audio_features(audio_mel).flatten(1)
        h_a = self.audio_drop(F.relu(self.audio_proj(x)))
        return self.classifier(torch.cat([h_v, h_a], dim=-1))


def load_bimodal() -> BiLSTMBimodal:
    """Load the per-fold bimodal head trained by iemocap_benchmark_4class.ipynb.
    The benchmark saves through a `BimodalModel(head)` wrapper, so every key
    in the checkpoint is prefixed with `head.` -- we strip that prefix here."""
    model = BiLSTMBimodal(feat_dim_v=FEAT_EFF, hidden=128, n_classes=N_CLS,
                          n_mels=N_MELS, mel_len=MAX_LEN).to(DEVICE)
    ckpt_path = BIMODAL_CKPT_DIR / f"ours_bilstm_bimodal_hard_skip_{BIMODAL_SESSION}.pt"
    raw = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    stripped = {k[len("head."):]: v for k, v in raw.items() if k.startswith("head.")}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    assert not missing,    f"missing keys when loading bimodal head: {missing}"
    assert not unexpected, f"unexpected keys when loading bimodal head: {unexpected}"
    model.eval()
    return model
