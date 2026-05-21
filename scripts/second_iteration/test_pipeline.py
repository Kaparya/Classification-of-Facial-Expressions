#!/usr/bin/env python3
"""Smoke-tests for the second-iteration emotion pipeline.

Tests:
  1. Config paths exist
  2. Models load without error
  3. audio_to_mel produces the correct tensor shape
  4. Pipeline.predict on synthetic data (random audio + black frames)
  5. Pipeline.predict on a real IEMOCAP wav (if available)

Run:
    source venv/bin/activate && python scripts/second_iteration/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from config import (BIMODAL_CKPT_DIR, BIMODAL_SESSION, FACE_DETECTOR_PATH,
                    SR, N_MELS, MAX_LEN, N_FRAMES, LABELS, DEVICE)

IEMOCAP_SAMPLE = (
    Path(__file__).parent.parent.parent
    / "datasets" / "IEMOCAP_full_release"
    / "Session1" / "sentences" / "wav"
    / "Ses01F_impro01" / "Ses01F_impro01_F000.wav"
)


def section(title: str) -> None:
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print('─' * 50)


def test_paths() -> None:
    section("1. Config paths")
    bimodal_ckpt = BIMODAL_CKPT_DIR / f"ours_bilstm_bimodal_hard_skip_{BIMODAL_SESSION}.pt"
    gate_ckpt    = BIMODAL_CKPT_DIR / f"gate_iemocap_{BIMODAL_SESSION}.pt"
    for name, p in [
        ("bimodal head ckpt", bimodal_ckpt),
        ("gate ckpt",         gate_ckpt),
        ("face detector",     FACE_DETECTOR_PATH),
    ]:
        assert p.exists(), f"MISSING: {p}"
        print(f"  OK  {name}: {p.name}")
    print(f"  Device: {DEVICE}")


def test_mel_shape() -> None:
    section("2. audio_to_mel tensor shape")
    from models.mel_cnn import audio_to_mel
    dummy = np.zeros(SR * 2, dtype=np.float32)
    t = audio_to_mel(dummy)
    assert t.shape == (1, 1, N_MELS, MAX_LEN), f"Unexpected shape: {t.shape}"
    print(f"  OK  shape {tuple(t.shape)}")


def test_model_load() -> None:
    section("3. Model loading")
    from models.mel_cnn import load_mel_gate
    gate = load_mel_gate()
    print(f"  OK  MelCNN          params={sum(p.numel() for p in gate.parameters()):,}")

    from models.bimodal import load_bimodal
    head = load_bimodal()
    print(f"  OK  BiLSTMBimodal   params={sum(p.numel() for p in head.parameters()):,}")


def test_synthetic_inference() -> None:
    section("4. Synthetic inference (random audio + black frames)")
    from models.pipeline import EmotionPipeline
    pipeline = EmotionPipeline()

    audio  = np.random.randn(SR * 2).astype(np.float32) * 0.01
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * N_FRAMES

    label, conf = pipeline.predict(audio, frames)
    assert label in LABELS, f"Unknown label: {label}"
    assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"
    print(f"  OK  result: {label!r}  conf={conf:.2%}")


def test_real_sample() -> None:
    section("5. Real IEMOCAP sample (optional)")
    if not IEMOCAP_SAMPLE.exists():
        print(f"  SKIP  sample not found: {IEMOCAP_SAMPLE}")
        return

    import librosa
    from models.pipeline import EmotionPipeline

    y, _ = librosa.load(IEMOCAP_SAMPLE, sr=SR, mono=True)
    frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * N_FRAMES

    pipeline = EmotionPipeline()
    label, conf = pipeline.predict(y, frames)
    print(f"  OK  file: {IEMOCAP_SAMPLE.name}")
    print(f"       result: {label!r}  conf={conf:.2%}")


if __name__ == '__main__':
    test_paths()
    test_mel_shape()
    test_model_load()
    test_synthetic_inference()
    test_real_sample()
    print("\n✓ All tests passed.\n")
