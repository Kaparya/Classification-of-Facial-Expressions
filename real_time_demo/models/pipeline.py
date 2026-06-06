import numpy as np
import torch
import torch.nn.functional as F
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import torchvision.transforms as T
from emotiefflib.facial_analysis import EmotiEffLibRecognizer

from config import DEVICE, FACE_DETECTOR_PATH, IMG_SIZE, N_FRAMES, LABELS
from models.mel_cnn import load_mel_gate, audio_to_mel
from models.bimodal import load_bimodal

_IMAGENET_TF = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EmotionPipeline:
    """Hard cascade matching iemocap_benchmark_4class.ipynb (cascade_mode='hard').

    Stage 1 — per-fold IEMOCAP MelCNN gate on the (1,1,64,128) log-mel.
      gate.argmax == 0 (neutral)  -> return 'neutral' immediately;
                                     face detection, EmotiEffNet, and the
                                     bimodal head are skipped entirely.
      gate.argmax == 1 (non-neu)  -> run stage 2.

    Stage 2 — bimodal head on (video_eff, audio_mel). The same log-mel from
    stage 1 is reused, so audio is never re-extracted. The head's argmax
    over 4 classes is the final prediction (no log-prior fusion -- the gate
    only routes; it does NOT bias the head's logits)."""

    def __init__(self):
        print("Loading audio gate (MelCNN, per-fold IEMOCAP gate)...")
        self.mel_gate = load_mel_gate()

        print("Loading video backbone (EmotiEffNet enet_b0_8_best_vgaf)...")
        _rec = EmotiEffLibRecognizer(engine='torch', model_name='enet_b0_8_best_vgaf',
                                     device=str(DEVICE))
        self.emotieffnet = _rec.model.to(DEVICE).eval()
        for p in self.emotieffnet.parameters():
            p.requires_grad_(False)

        print("Loading bimodal head (BiLSTMBimodal, 4 classes, hard skip)...")
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
        gate_probs  = F.softmax(gate_logits, dim=1)[0]

        # Hard cascade: gate says neutral -> short-circuit, skip video pipeline.
        if gate_logits.argmax(dim=1).item() == 0:
            return 'neutral', gate_probs[0].item()

        # Gate said non-neutral. Fall back to gate-only neutral if camera dropped.
        if not frames:
            return 'neutral', gate_probs[0].item()

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

        probs = F.softmax(head_logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        return LABELS[pred_idx], probs[pred_idx].item()
