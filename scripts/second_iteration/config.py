from pathlib import Path
import torch

ROOT = Path(__file__).parent.parent.parent  # Diploma/

AUDIO_CKPT_DIR  = ROOT / "trained_models" / "audio_neutral_3_augmentation"
IEMOCAP_CKPT_DIR = ROOT / "trained_models" / "iemocap_benchmark"
FACE_DETECTOR_PATH = ROOT / "raw_models" / "blaze_face_short_range.tflite"

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

# IEMOCAP 6-class label space
N_CLS  = 6
LABELS = ['neutral', 'happy', 'sad', 'angry', 'excited', 'frustrated']

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else
    ('mps' if torch.backends.mps.is_available() else 'cpu')
)

# VAD / streaming
VAD_CHUNK       = 512   # samples per Silero VAD call
VAD_THRESHOLD   = 0.5
SILENCE_PATIENCE = 12   # silent chunks before ending an utterance
