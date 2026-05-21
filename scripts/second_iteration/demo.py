#!/usr/bin/env python3
"""Real-time multimodal emotion recognition — second iteration.

Hard cascade pipeline (final shipped form from iemocap_benchmark_4class.ipynb,
cascade_mode='hard'; per-fold LOSO IEMOCAP checkpoints).
  1. MelCNN audio gate (per-fold gate_iemocap_{BIMODAL_SESSION}.pt).
     → neutral?      output 'neutral' immediately; face detection,
                     EmotiEffNet, and the bimodal head are SKIPPED.
     → non-neutral?  run stage 2.
  2. EmotiEffNet backbone (enet_b0_8_best_vgaf) + BiLSTMBimodal head
     (ours_bilstm_bimodal_hard_skip_{BIMODAL_SESSION}.pt):
     video BiLSTM late-fused with the gate's conv-stack audio branch — the
     (1,1,64,128) log-mel from stage 1 is reused, no audio re-extraction.
     argmax over 4 IEMOCAP classes: neutral / happy / sad / angry.

Usage:
    source venv/bin/activate && python scripts/second_iteration/demo.py

Press Q to quit.
"""

import sys
import threading
from pathlib import Path

import cv2
import numpy as np

# Put this package on sys.path so sub-modules can do `from config import …`
sys.path.insert(0, str(Path(__file__).parent))

from capture.audio_stream import AudioStream
from capture.video_stream import VideoStream
from models.pipeline import EmotionPipeline
from display.overlay import draw_overlay


def main() -> None:
    print("=== Multimodal Emotion Recognition — Second Iteration ===\n")

    pipeline     = EmotionPipeline()
    audio_stream = AudioStream()
    video_stream = VideoStream(camera_idx=0, buffer_seconds=3.0)

    audio_stream.start()
    video_stream.start()

    result_lock   = threading.Lock()
    emotion_label = 'neutral'
    emotion_conf  = 1.0
    processing    = False

    def run_inference(audio: np.ndarray, frames: list) -> None:
        nonlocal emotion_label, emotion_conf, processing
        try:
            label, conf = pipeline.predict(audio, frames)
            with result_lock:
                emotion_label = label
                emotion_conf  = conf
            print(f"  → {label.upper():<11}  conf {conf:.0%}")
        except Exception as exc:
            print(f"[inference error] {exc}")
        finally:
            processing = False

    print("\nSpeak into the microphone to trigger emotion recognition.")
    print("Press Q in the video window to quit.\n")

    try:
        while True:
            audio = audio_stream.get_utterance()
            if audio is not None and not processing:
                frames = video_stream.get_buffer()
                processing = True
                t = threading.Thread(target=run_inference, args=(audio, frames), daemon=True)
                t.start()

            frame = video_stream.get_latest_frame()
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            with result_lock:
                label = emotion_label
                conf  = emotion_conf

            out = draw_overlay(frame_bgr, label, conf, listening=audio_stream.is_listening)
            cv2.imshow('Emotion Recognition — Second Iteration', out)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        audio_stream.stop()
        video_stream.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == '__main__':
    main()
