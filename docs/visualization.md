# Visualization & Diagnostics

TensorBoard Embeddings
- Consolidator can log PCA(2) of memory keys into TensorBoard’s projector.
- Pass a SummaryWriter and a log cadence to `buffer.consolidate` or directly to `Consolidator.sleep`.

Example
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/hippotorch")

# inside training loop
metrics = buffer.consolidate(steps=100, tb_writer=writer, tb_log_every=1000)
```

PCA utilities
```python
from hippotorch import pca_keys, pca_keys_with_rewards
comps, coords = pca_keys(memory.keys.cpu(), k=2)
```

Corridor Regret Plot
- Run multi-seed and plot net improvement (delta) of hybrid vs uniform.
```bash
python -m examples.corridor_multiseed --seeds 5 --episodes 800 --length 30 \
  --coin-reward 0.0 --mixture 0.5 --schedule-start 0.0 --schedule-end 0.5 \
  --warmup-episodes 600 --reward-weight 0.9 --temperature 0.05 \
  --cons-every 5 --cons-steps 200 --out results/corridor_multiseed.csv
python scripts/plot_corridor_regret.py --csv results/corridor_multiseed.csv --out results/corridor_regret.png --json results/corridor_regret.json
```

