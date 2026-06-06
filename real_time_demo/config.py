from pathlib import Path
import torch

ROOT = Path(__file__).parent.parent.parent  # Diploma/

AUDIO_CKPT_DIR  = ROOT / "trained_models" / "audio_neutral_3_augmentation"
BIMODAL_CKPT_DIR = ROOT / "trained_models" / "iemocap_benchmark_4class" / "checkpoints"
FACE_DETECTOR_PATH = ROOT / "raw_models" / "blaze_face_short_range.tflite"

# Per-fold checkpoint selection. Picked as the highest test-UA LOSO fold of
# `ours_bilstm_bimodal_hard_skip_scratch` in
# results/ours_bilstm_bimodal_hard_skip_scratch/fold_*.json:
#   Session1=0.5419  Session2=0.4704  Session3=0.5366
#   Session4=0.5243  Session5=0.5385  -> Session1 wins.
BIMODAL_SESSION = "Session1"

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

# VAD / streaming
VAD_CHUNK       = 512   # samples per Silero VAD call
VAD_THRESHOLD   = 0.5
SILENCE_PATIENCE = 12   # silent chunks before ending an utterance

