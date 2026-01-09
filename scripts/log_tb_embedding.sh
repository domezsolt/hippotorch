#!/usr/bin/env bash
set -euo pipefail

# Log a one-off TensorBoard embedding snapshot to runs/hippo_tb

python - <<'PY'
import torch
from torch.utils.tensorboard import SummaryWriter
from hippotorch import DualEncoder, MemoryStore, Consolidator
from hippotorch.core.episode import Episode
writer=SummaryWriter('runs/hippo_tb')
enc=DualEncoder(input_dim=5, embed_dim=16)
mem=MemoryStore(embed_dim=16)
for i in range(100):
    s=torch.randn(6,3); a=torch.zeros(6,1); r=torch.randn(6)
    ep=Episode(s,a,r)
    x=torch.cat([s,a,r.unsqueeze(-1)],-1).unsqueeze(0)
    mem.write(enc.encode_key(x),[ep],encoder_step=i)
con=Consolidator(enc, temperature=0.07, reward_weight=0.8)
con.sleep(mem, steps=50, batch_size=64, tb_writer=writer, tb_log_every=10)
print('logged to runs/hippo_tb')
PY

