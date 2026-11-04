import sys, numpy as np, MDAnalysis as mda, warnings; warnings.filterwarnings("ignore")
top, traj = sys.argv[1], (sys.argv[2] if len(sys.argv)>2 else None)
u = mda.Universe(top, traj) if traj else mda.Universe(top)
prot = u.select_atoms("protein")
res = prot.residues
rows=[]
for i in range(1,len(res)):
    r, p = res[i], res[i-1]
    if r.resname=="PRO": continue
    N=r.atoms.select_atoms("name N"); H=r.atoms.select_atoms("name H or name HN"); CA=r.atoms.select_atoms("name CA"); C=p.atoms.select_atoms("name C")
    if len(N)!=1 or len(H)!=1 or len(CA)!=1 or len(C)!=1: rows.append((r.resid, None)); continue
    rows.append((r.resid,(N[0].index,H[0].index,CA[0].index,C[0].index)))
nH=sum(1 for _,x in rows if x is None); print(f"{top} {traj or ''}: residues checked {len(rows)}, missing N/H/CA/Cprev: {nH}")
idx=np.array([x for _,x in rows if x]); resids=[r for r,x in rows if x]
L=[];ANG=[];OOP=[]
for ts in u.trajectory:
    X=u.atoms.positions
    N,H,CA,C=X[idx[:,0]],X[idx[:,1]],X[idx[:,2]],X[idx[:,3]]
    nh=H-N; l=np.linalg.norm(nh,axis=1); nh/=l[:,None]
    a=(CA-N); a/=np.linalg.norm(a,axis=1)[:,None]; b=(C-N); b/=np.linalg.norm(b,axis=1)[:,None]
    bis=-(a+b); bis/=np.linalg.norm(bis,axis=1)[:,None]        # ideal H direction: opposite the bisector (outward)
    ang=np.degrees(np.arccos(np.clip((nh*bis).sum(1),-1,1)))
    nrm=np.cross(a,b); nrm/=np.linalg.norm(nrm,axis=1)[:,None]; oop=np.degrees(np.arcsin(np.abs((nh*nrm).sum(1))))
    L.append(l);ANG.append(ang);OOP.append(oop)
L=np.array(L);ANG=np.array(ANG);OOP=np.array(OOP)
print(f"frames {L.shape[0]}  N-H length: mean {L.mean():.3f} min {L.min():.3f} max {L.max():.3f}")
print(f"angle(N-H, outward anti-bisector): mean {ANG.mean():.1f} deg, 95th pct {np.percentile(ANG,95):.1f}, max {ANG.max():.1f}; frac >30deg {np.mean(ANG>30):.4f}, frac >90deg (inward) {np.mean(ANG>90):.5f}")
print(f"out-of-plane: mean {OOP.mean():.1f} deg, 95th pct {np.percentile(OOP,95):.1f}, max {OOP.max():.1f}")
bad=np.argwhere(ANG>30)
if len(bad): 
    import collections; c=collections.Counter(resids[j] for _,j in bad); print("residues with angle>30 (count over frames):", sorted(c.items())[:20])
    print("frames with any angle>30:", len(set(bad[:,0])))
