import torch
import torch.nn as nn
import numpy as np
import librosa

from config import (BIMODAL_CKPT_DIR, BIMODAL_SESSION,
                    SR, N_MELS, MAX_LEN, HOP, MEL_MEAN, MEL_STD, DEVICE)


class MelCNN(nn.Module):
    def __init__(self, n_mels: int = 64, max_len: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (n_mels // 8) * (max_len // 8), 256),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_mel_gate() -> MelCNN:
    """Load the per-LOSO-fold IEMOCAP-fine-tuned MelCNN gate. The global
    RAVDESS+CREMA-D pretrained gate over-fires `neutral` on IEMOCAP
    improvised speech; this fine-tuned copy is what
    iemocap_benchmark_4class.ipynb uses for the cascade prior."""
    model = MelCNN(N_MELS, MAX_LEN).to(DEVICE)
    state = torch.load(BIMODAL_CKPT_DIR / f"gate_iemocap_{BIMODAL_SESSION}.pt",
                       map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def audio_to_mel(y: np.ndarray) -> torch.Tensor:
    """float32 waveform → (1, 1, N_MELS, MAX_LEN) tensor ready for MelCNN."""
    mel = librosa.feature.melspectrogram(y=y.astype(np.float32), sr=SR, n_mels=N_MELS, hop_length=HOP)
    lm = librosa.power_to_db(mel, ref=np.max)
    if lm.shape[1] < MAX_LEN:
        lm = np.pad(lm, ((0, 0), (0, MAX_LEN - lm.shape[1])))
    else:
        lm = lm[:, :MAX_LEN]
    lm = (lm - MEL_MEAN) / (MEL_STD + 1e-8)
    return torch.from_numpy(lm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
