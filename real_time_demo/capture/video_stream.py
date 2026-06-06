import collections
import threading
import cv2
import numpy as np


class VideoStream:
    """Captures webcam frames in a background thread with a rolling RGB buffer."""

    def __init__(self, camera_idx: int = 0, buffer_seconds: float = 3.0, fps: int = 15):
        self._camera_idx = camera_idx
        self._fps = fps
        self._buffer: collections.deque = collections.deque(maxlen=int(buffer_seconds * fps))
        self._lock = threading.Lock()
        self._running = False
        self._latest_frame: np.ndarray | None = None
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._camera_idx)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self._camera_idx}")
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"Video capture started  (camera {self._camera_idx}, ~{self._fps} fps)")

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._buffer.append(frame_rgb)
                self._latest_frame = frame_rgb

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_buffer(self) -> list[np.ndarray]:
        with self._lock:
            return list(self._buffer)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
