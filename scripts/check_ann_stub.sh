#!/usr/bin/env bash
set -euo pipefail

# Quick ANN stub retrieval check (uses exact matmul fallback)

python - <<'PY'
import torch
from hippotorch.utils.ann import ApproximateNearestNeighborIndex
from hippotorch import MemoryStore, DualEncoder
from hippotorch.core.episode import Episode
enc=DualEncoder(input_dim=5, embed_dim=16)
index=ApproximateNearestNeighborIndex()
mem=MemoryStore(embed_dim=16, indexer=index)
for i in range(500):
    s=torch.randn(4,3); a=torch.zeros(4,1); r=torch.randn(4); ep=Episode(s,a,r)
    x=torch.cat([s,a,r.unsqueeze(-1)],-1).unsqueeze(0)
    mem.write(enc.encode_key(x),[ep],encoder_step=i)
q=torch.randn(1,16)
scores, eps = mem.retrieve(q, top_k=5, encoder=enc, current_step=0, lazy_refresh=False)
print(scores.shape, len(eps[0]))
PY

