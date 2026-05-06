import cv2
import numpy as np

# BGR colours for each emotion
_COLOURS: dict[str, tuple[int, int, int]] = {
    'neutral':    (180, 180, 180),
    'happy':      (80,  200,   0),
    'sad':        (255, 120,  80),
    'angry':      (60,   60, 255),
    'excited':    (0,   180, 255),
    'frustrated': (200, 100, 255),
}
_DEFAULT_COLOUR = (200, 200, 200)


def draw_overlay(frame_bgr: np.ndarray, emotion: str, confidence: float,
                 listening: bool) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    colour = _COLOURS.get(emotion, _DEFAULT_COLOUR)

    # Semi-transparent dark bar at the bottom
    bar_h = 72
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

    # Emotion label + confidence
    label_txt = f"{emotion.upper()}  {confidence:.0%}"
    cv2.putText(out, label_txt, (14, h - bar_h + 36),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, colour, 2, cv2.LINE_AA)

    # Listening / idle indicator
    status_txt = "LISTENING..." if listening else "IDLE"
    status_col = (0, 220, 80) if listening else (110, 110, 110)
    cv2.putText(out, status_txt, (14, h - bar_h + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 1, cv2.LINE_AA)

    # Thin confidence bar along the bottom edge
    bar_w = max(0, int((w - 28) * confidence))
    cv2.rectangle(out, (14, h - 5), (14 + bar_w, h - 1), colour, -1)

    return out
