# trained_models/

Trained weights and experiment artifacts. Current directories live at the root; superseded ones are under `old/`.

## Current

### `audio_neutral_3_augmentation/`
Binary "neutral / non-neutral" audio classifier (RAVDESS + CREMA-D, trained with offline augmentation). Source of the cascade's MelCNN gate.
- `mel_cnn.pt` — Log-Mel + CNN (used as the first-stage cascade gate).
- `linear_classifier.pt` — DistilHuBERT + Linear.
- `wav2vec2_finetuned.pt`, `wav2vec2_head.pt` — Wav2Vec2 + Linear.
- `svm.pkl`, `svm_scaler.pkl` — MFCC + SVM.
- `meta.json` — metrics + `mel_mean`/`mel_std` for log-mel normalization at inference.

### `video_emotion_3_augmentation/`
Video emotion classifier (RAVDESS + CREMA-D, offline augmentation). Two backbones × three heads.
- `emotieffnet_b0_backbone.pt`, `mobilenet_v3_small_backbone.pt` — backbones.
- `head_emotieffnet_{bigru,bilstm,tcn}.pt`, `head_mobilenet_{bigru,bilstm,tcn}.pt` — classifier heads. `head_emotieffnet_bilstm.pt` is the pretrained init for the cascade's video branch.
- `meta.json` — config and metrics.

### `iemocap_benchmark_4class/`
Outputs of the main benchmark (`notebooks/iemocap_benchmark_4class.ipynb`): 4 classes (`neutral, happy∪excited, sad, angry`), LOSO 5-fold. Final architecture is `ours_bilstm_bimodal_hard_skip_scratch`.
- `checkpoints/` — per-fold weights (`{slug}_Session{1..5}.pt`):
  - `gate_iemocap_*` — MelCNN gate fine-tuned on IEMOCAP per fold (shared across all cascade models within a fold).
  - `ours_bilstm_bimodal_hard_skip_scratch_*` — **final model** (bimodal head, scratch init, hard cascade).
  - `ours_bilstm_bimodal_hard_skip_*` — same variant with pretrained init (ablation control).
  - `gia_mic_*`, `mm_nodeformer_*`, `scme_gf_*`, `resnet18_gru_*` — competitors.
- `logs/training_curves.png` — training curves.
- `run_meta.json` — run metadata. Per-fold metrics live in the root-level `results/{slug}/fold_*.json`.

## `old/` — superseded (not used)

- `audio_neutral_0/`, `audio_neutral_1/`, `audio_neutral_2_base/` — earlier audio-gate versions (no / different augmentation).
- `video_emotion_0/`, `video_emotion_1/`, `video_emotion_2_base/` — earlier video-classifier versions (incl. EfficientNet backbone and attention/transformer heads).
- `iemocap_benchmark/` — old 6-class IEMOCAP benchmark (single split): fine-tuned heads, competitors, `benchmark_results.csv`, `eval_results/`.
- `iemocap_benchmark_4class_base/` — previous version of the 4-class benchmark.
- `iemocap_e2e_multimodal/` — 4-class LOSO cascade from `iemocap_e2e_multimodal.ipynb`: `folds/`, `models/` (per-fold gate+video), `loso_results.pkl`, `results_table.csv`, `confusion_matrix.png`.
