#!/usr/bin/env bash
set -euo pipefail

# Compare temporal-only vs reward-weighted consolidation on clustering metrics

python - <<'PY'
import torch
from hippotorch import DualEncoder, MemoryStore, Consolidator, Episode

def make_ep(T=8, sd=3, ad=1, sv=1.0, rv=1.0):
    s=torch.full((T,sd),sv); a=torch.zeros(T,ad); r=torch.full((T,),rv); return Episode(s,a,r)

sd,ad=3,1; inp=sd+ad+1; emb=32
enc=DualEncoder(input_dim=inp, embed_dim=emb)
mem=MemoryStore(embed_dim=emb, capacity=200)
for i,rv in enumerate(([1.0]*80+[0.1]*120)):
    ep=make_ep(sv=(1.0 if i<80 else -1.0), rv=rv)
    x=torch.cat([ep.states,ep.actions,ep.rewards.unsqueeze(-1)],-1).unsqueeze(0)
    mem.write(enc.encode_key(x),[ep],encoder_step=i)

def run(weighted):
    con=Consolidator(enc, temperature=0.05, reward_weight=(0.8 if weighted else 0.0))
    con.sleep(mem, steps=300, batch_size=128, refresh_keys=True)
    return con.get_representation_quality(mem)
print('temporal_only', run(False))
print('reward_weighted', run(True))
PY
