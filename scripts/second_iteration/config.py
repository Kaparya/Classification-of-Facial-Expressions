from pathlib import Path
import torch

ROOT = Path(__file__).parent.parent.parent  # Diploma/

AUDIO_CKPT_DIR  = ROOT / "trained_models" / "audio_neutral_3_augmentation"
BIMODAL_CKPT_DIR = ROOT / "trained_models" / "iemocap_benchmark_4class" / "checkpoints"
FACE_DETECTOR_PATH = ROOT / "raw_models" / "blaze_face_short_range.tflite"

# Per-fold checkpoint selection: Session5 is the last LOSO fold (conventional
# default in the benchmark's summary plots). Swap to a different session if a
# specific fold's UA peaks higher on improvised speech.
BIMODAL_SESSION = "Session5"

# Audio (matches audio_neutral_3_augmentation/meta.json)
SR      = 16_000
N_MELS  = 64
MAX_LEN = 128
HOP     = 512
MEL_MEAN = -31.409805297851562
MEL_STD  = 26.3829288482666

# Video
N_FRAMES = 16
IMG_SIZE = 224
FEAT_EFF = 1_280  # EmotiEffNet output dim

# IEMOCAP 4-class label space (matches iemocap_benchmark_4class.ipynb:
# excited folded into happy; frustrated/fear/surprise/disgust/oth/xxx dropped).
N_CLS  = 4
LABELS = ['neutral', 'happy', 'sad', 'angry']

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else
    ('mps' if torch.backends.mps.is_available() else 'cpu')
)

# Cascade fusion (matches iemocap_benchmark_4class.ipynb SOFT_LAM):
# final_logits[0]   = head_logits[0]   + SOFT_LAM * log p_gate(neutral)
# final_logits[k>0] = head_logits[k]   + SOFT_LAM * log p_gate(non-neutral)
# lam=1.0 -> Bayesian product of head and gate; same value as the benchmark.
SOFT_LAM = 1.0

# VAD / streaming
VAD_CHUNK       = 512   # samples per Silero VAD call
VAD_THRESHOLD   = 0.5
SILENCE_PATIENCE = 12   # silent chunks before ending an utterance
