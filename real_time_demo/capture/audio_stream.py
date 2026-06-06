import threading
import numpy as np
import torch
import sounddevice as sd

from config import SR, VAD_CHUNK, VAD_THRESHOLD, SILENCE_PATIENCE


class AudioStream:
    """Captures microphone audio, applies Silero VAD, and exposes completed utterances."""

    def __init__(self, vad_threshold: float = VAD_THRESHOLD,
                 silence_patience: int = SILENCE_PATIENCE):
        self._vad_threshold = vad_threshold
        self._silence_patience = silence_patience
        self._vad_model = None
        self._buffer: list[np.ndarray] = []
        self._speaking = False
        self._silent_count = 0
        self._lock = threading.Lock()
        self._ready_audio: np.ndarray | None = None
        self._stream: sd.InputStream | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        print("Loading Silero VAD from torch hub...")
        self._vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            verbose=False,
        )
        self._vad_model.eval()

        self._stream = sd.InputStream(
            samplerate=SR,
            channels=1,
            dtype='float32',
            blocksize=VAD_CHUNK,
            callback=self._callback,
        )
        self._stream.start()
        print(f"Audio capture started  ({SR} Hz, chunk {VAD_CHUNK} samples)")

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        chunk = indata[:, 0].copy()
        t = torch.from_numpy(chunk)
        with torch.no_grad():
            vad_prob: float = self._vad_model(t, SR).item()
        is_speech = vad_prob > self._vad_threshold

        with self._lock:
            if is_speech:
                self._buffer.append(chunk)
                self._speaking = True
                self._silent_count = 0
            elif self._speaking:
                self._buffer.append(chunk)
                self._silent_count += 1
                if self._silent_count >= self._silence_patience:
                    self._ready_audio = np.concatenate(self._buffer)
                    self._buffer = []
                    self._speaking = False
                    self._silent_count = 0

    # ------------------------------------------------------------------
    def get_utterance(self) -> np.ndarray | None:
        """Pop and return the latest completed utterance, or None."""
        with self._lock:
            audio = self._ready_audio
            self._ready_audio = None
        return audio

    @property
    def is_listening(self) -> bool:
        with self._lock:
            return self._speaking

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
