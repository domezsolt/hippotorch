#!/usr/bin/env bash
set -euo pipefail

# Consolidation micro-benchmark: reward alignment and retrieval sharpness

python - <<'PY'
import torch
from hippotorch import DualEncoder, MemoryStore, Consolidator
from hippotorch.core.episode import Episode

def make_ep(T=8, sd=3, ad=1, sv=1.0, rv=1.0):
    s=torch.full((T,sd),sv); a=torch.zeros(T,ad); r=torch.full((T,),rv); return Episode(s,a,r)

sd,ad=3,1; inp=sd+ad+1; emb=32
enc=DualEncoder(input_dim=inp, embed_dim=emb)
mem=MemoryStore(embed_dim=emb, capacity=200)
# positives/high-reward and distractors/low-reward
eps=[make_ep(sv=1.0,rv=1.0) for _ in range(80)]+[make_ep(sv=-1.0,rv=0.1) for _ in range(120)]
for i,ep in enumerate(eps):
    x=torch.cat([ep.states,ep.actions,ep.rewards.unsqueeze(-1)],-1).unsqueeze(0)
    k=enc.encode_key(x); mem.write(k,[ep],encoder_step=i)
con=Consolidator(enc, temperature=0.05, reward_weight=0.8, learning_rate=1e-4)
print('before', con.get_representation_quality(mem))
print('sleep', con.sleep(mem, steps=400, batch_size=128, refresh_keys=True, report_quality=True))
print('after', con.get_representation_quality(mem))
PY

