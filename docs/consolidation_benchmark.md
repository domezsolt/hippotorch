# Consolidation Benchmark

This script measures improvements in representation quality (reward alignment and retrieval sharpness) before and after consolidation on synthetic episodes.

Run
```bash
python -m examples.consolidation_clustering --steps 200 --batch-size 64
```

Outputs
- `before`: `retrieval_sharpness`, `reward_alignment`, `key_dispersion`, `avg_staleness`
- `consolidation`: loss metrics during sleep
- `after`: same diagnostics post-consolidation

Expected behavior
- `reward_alignment` increases after consolidation on this synthetic task.

Key file
- `examples/consolidation_clustering.py`

