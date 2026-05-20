import numpy as np
import torch
import torch.nn.functional as F
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import torchvision.transforms as T
from emotiefflib.facial_analysis import EmotiEffLibRecognizer

from config import DEVICE, FACE_DETECTOR_PATH, IMG_SIZE, N_FRAMES, LABELS, SOFT_LAM
from models.mel_cnn import load_mel_gate, audio_to_mel
from models.bimodal import load_bimodal

_IMAGENET_TF = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EmotionPipeline:
    """Soft cascade matching iemocap_benchmark_4class.ipynb. Gate and head both
    run on every utterance; gate's log_softmax is added to head logits as a
    Bayesian prior (SOFT_LAM). Audio_mel is shared between gate and head -- no
    re-extraction. Same fusion as the metrics reported in the benchmark."""

    def __init__(self):
        print("Loading audio gate (MelCNN, per-fold IEMOCAP gate)...")
        self.mel_gate = load_mel_gate()

        print("Loading video backbone (EmotiEffNet enet_b0_8_best_vgaf)...")
        _rec = EmotiEffLibRecognizer(engine='torch', model_name='enet_b0_8_best_vgaf',
                                     device=str(DEVICE))
        self.emotieffnet = _rec.model.to(DEVICE).eval()
        for p in self.emotieffnet.parameters():
            p.requires_grad_(False)

        print("Loading bimodal head (BiLSTMBimodal, 4 classes)...")
        self.bimodal_head = load_bimodal()

        print("Loading face detector (MediaPipe BlazeFace)...")
        self.face_detector = mp_vision.FaceDetector.create_from_options(
            mp_vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(FACE_DETECTOR_PATH)),
                min_detection_confidence=0.5,
            )
        )
        print("Pipeline ready.")

    def _detect_face_crop(self, frame_rgb: np.ndarray) -> np.ndarray:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = self.face_detector.detect(mp_img)
        if res.detections:
            bb = res.detections[0].bounding_box
            h, w = frame_rgb.shape[:2]
            px = int(bb.width * 0.1)
            py = int(bb.height * 0.1)
            x1 = max(0, bb.origin_x - px)
            y1 = max(0, bb.origin_y - py)
            x2 = min(w, bb.origin_x + bb.width + px)
            y2 = min(h, bb.origin_y + bb.height + py)
            if x2 > x1 and y2 > y1:
                return cv2.resize(frame_rgb[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
        # fallback: centre crop
        h, w = frame_rgb.shape[:2]
        s = min(h, w)
        y1, x1 = (h - s) // 2, (w - s) // 2
        return cv2.resize(frame_rgb[y1:y1 + s, x1:x1 + s], (IMG_SIZE, IMG_SIZE))

    @torch.no_grad()
    def predict(self, audio: np.ndarray, frames: list) -> tuple[str, float]:
        """
        audio  : float32 numpy array at 16 kHz
        frames : list of RGB uint8 ndarrays (any length)
        Returns: (emotion_label, confidence_float)
        """
        mel = audio_to_mel(audio).to(DEVICE)
        gate_logits = self.mel_gate(mel)                       # (1, 2)
        gate_logp = F.log_softmax(gate_logits, dim=1)

        if not frames:
            # Camera unavailable: degrade gracefully to gate-only neutral check.
            p_neu = F.softmax(gate_logits, dim=1)[0, 0].item()
            return 'neutral', p_neu

        n = len(frames)
        if n >= N_FRAMES:
            idx = np.linspace(0, n - 1, N_FRAMES, dtype=int)
            sampled = [frames[i] for i in idx]
        else:
            sampled = frames + [frames[-1]] * (N_FRAMES - n)

        crops = [self._detect_face_crop(f) for f in sampled]
        batch = torch.stack([_IMAGENET_TF(c) for c in crops]).to(DEVICE)
        eff_feats = self.emotieffnet(batch)                    # (N_FRAMES, 1280)
        head_logits = self.bimodal_head(eff_feats.unsqueeze(0), mel)  # (1, N_CLS)

        # Soft cascade: add gate's log-prior to head logits.
        fused = head_logits.clone()
        fused[:, 0]  = fused[:, 0]  + SOFT_LAM * gate_logp[:, 0]
        fused[:, 1:] = fused[:, 1:] + SOFT_LAM * gate_logp[:, 1:2]

        probs = F.softmax(fused, dim=1)[0]
        pred_idx = probs.argmax().item()
        return LABELS[pred_idx], probs[pred_idx].item()
