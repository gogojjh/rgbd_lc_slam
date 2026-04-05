import numpy as np
from pathlib import Path

def load(p: str):
    a=[]
    for line in Path(p).read_text().splitlines():
        if not line.strip() or line[0]=='#':
            continue
        vals=list(map(float,line.split()))
        xyz=vals[1:4]
        a.append(xyz)
    return np.array(a)

est=load('results/runs/icl_baseline_evo/living_room_traj0_frei_png/traj_est_tum.txt')
gt=load('results/runs/icl_baseline_evo/living_room_traj0_frei_png/traj_gt_tum.txt')
print('est shape', est.shape)
print('gt shape', gt.shape)
print('est range', est.min(0), est.max(0))
print('gt range', gt.min(0), gt.max(0))
print('est std', est.std(0))
print('gt std', gt.std(0))
