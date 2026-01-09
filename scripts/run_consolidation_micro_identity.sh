#!/usr/bin/env bash
set -euo pipefail

# Consolidation micro-benchmark using IdentityDualEncoder (deterministic)

python - <<'PY'
import torch
from hippotorch import MemoryStore
from hippotorch.core.episode import Episode
from hippotorch.memory.consolidator import Consolidator
from hippotorch.memory.regression import IdentityDualEncoder

def make_ep(T=8, sd=3, ad=1, sv=1.0, rv=1.0):
    s=torch.full((T,sd),sv); a=torch.zeros(T,ad); r=torch.full((T,),rv); return Episode(s,a,r)

sd,ad=3,1; inp=sd+ad+1
enc=IdentityDualEncoder(input_dim=inp, refresh_interval=1)
mem=MemoryStore(embed_dim=inp, capacity=200)

eps=[make_ep(sv=1.0,rv=1.0) for _ in range(80)]+[make_ep(sv=-1.0,rv=0.1) for _ in range(120)]
for i,ep in enumerate(eps):
    x=torch.cat([ep.states,ep.actions,ep.rewards.unsqueeze(-1)],-1).unsqueeze(0)
    k=enc.encode_key(x); mem.write(k,[ep],encoder_step=i)

con=Consolidator(enc, temperature=0.05, reward_weight=0.8, learning_rate=5e-4)
print('before', con.get_representation_quality(mem))
print('sleep', con.sleep(mem, steps=400, batch_size=128, refresh_keys=True, report_quality=True))
print('after', con.get_representation_quality(mem))
PY

