#!/usr/bin/env bash
set -euo pipefail

# Verify POMDP window sampling shapes quickly

python - <<'PY'
import torch
from hippotorch import MemoryStore, HippocampalReplayBuffer
from hippotorch.memory.regression import IdentityDualEncoder
from hippotorch.core.episode import Episode
sd,ad=3,1; inp=sd+ad+1
enc=IdentityDualEncoder(input_dim=inp)
mem=MemoryStore(embed_dim=inp)
for v in [1.0, -1.0, 0.5]:
    s=torch.full((8,sd),v); a=torch.zeros(8,ad); r=torch.full((8,),v)
    ep=Episode(s,a,r)
    x=torch.cat([s,a,r.unsqueeze(-1)],-1).unsqueeze(0)
    mem.write(enc.encode_key(x),[ep])
buf=HippocampalReplayBuffer(mem, enc, mixture_ratio=0.5)
B=4
out=buf.sample_windows(batch_size=B, window_size=5, query_state=torch.cat([torch.ones(sd), torch.zeros(2)]), top_k=3)
print({k:tuple(v.shape) for k,v in out.items()})
PY

