# IEMOCAP MER Benchmark — 4-class, LOSO 5-fold

Audio + facial-expression emotion recognition benchmark on IEMOCAP, produced by
`notebooks/iemocap_benchmark_4class.ipynb`.

## Protocol
- **Classes:** neutral, happy (incl. excited), sad, angry.
- **Split:** Leave-One-Session-Out 5-fold; metrics are mean +- std over folds.
- **Metrics:** WA, UA/UAR (main), Macro-F1, per-class F1, confusion matrix.
- **Face selection:** gender-letter heuristic (speaker letter in the utterance
  id picks the half-frame); MediaPipe BlazeFace crop.

## Reproduce
1. `source venv/bin/activate && jupyter notebook`
2. Open `notebooks/iemocap_benchmark_4class.ipynb`, Run All. The first run
   builds the feature cache `datasets/iemocap_features_cache/` if missing
   (slow, one-time); later runs reuse it.
3. Outputs:
   - `metadata/iemocap_4class.csv` — dataset index
   - `results/<model>/fold_k.json`, `summary.json` — per-fold & aggregated metrics
   - `results/benchmark_table.csv` — final comparison table
   - `results/confusion_matrices.png`, `logs/training_curves.png`
   - `trained_models/iemocap_benchmark_4class/checkpoints/` — per-fold weights
   - `run_meta.json` — git hash, config, versions

Set `BENCH_QUICK=1` in the environment for a fast 2-epoch smoke run.
Reproducibility: fixed seeds, deterministic algorithms, git hash logged.
