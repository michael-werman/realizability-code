import numpy as np
rng=np.random.default_rng(2026)
F_PX, W = 800.0, 1000.0
def rand_rot():
    q=rng.normal(size=4); q/=np.linalg.norm(q); a,b,c,d=q
    return np.array([[a*a+b*b-c*c-d*d,2*(b*c-a*d),2*(b*d+a*c)],[2*(b*c+a*d),a*a-b*b+c*c-d*d,2*(c*d-a*b)],[2*(b*d-a*c),2*(c*d+a*b),a*a-b*b-c*c+d*d]])
def outline_conic(axes,R,T,f):
    # ellipsoid X = T + R diag(axes) u, |u|=1 ; camera at origin looking +z; image x = f X/Z (coords centered at p)
    A=R@np.diag(1/np.array(axes)**2)@R.T; s=T@A@T-1; u=A@T
    M=s*A-np.outer(u,u)          # tangent cone d^T M d = 0
    K=np.diag([f,f,1.0]); Ki=np.linalg.inv(K)
    return Ki.T@M@Ki             # image conic in pixel coords centered at p
def norm_sign(C):
    C=C/np.linalg.norm(C)
    A=C[:2,:2]
    if np.linalg.eigvalsh(A).min()<0: C=-C
    return C
def ellipse_params(C):
    A=C[:2,:2]; b=C[:2,2]; c=C[2,2]; ctr=-np.linalg.solve(A,b); k=-(c+b@ctr)
    ev,V=np.linalg.eigh(A); return ctr, np.sqrt(k/ev), V
def kappa(C,f):
    K=np.diag([f,f,1.0]); Q=K@C@K; ev=np.linalg.eigvalsh(Q)
    if (ev>0).sum()!=2: ev=-ev
    pos=np.sort(ev[ev>0]); return pos, ev
def fit(pts):
    x,y=pts[:,0],pts[:,1]; D=np.column_stack([x*x,x*y,y*y,x,y,np.ones_like(x)])
    _,_,Vt=np.linalg.svd(D); a,b,c,d,e,g=Vt[-1]
    return norm_sign(np.array([[a,b/2,d/2],[b/2,c,e/2],[d/2,e/2,g]]))
res={r:[] for r in (1.5,2.0,3.0)}
for ratio in res:
    n=0
    while n<200:
        c=1.0; a=ratio; b=np.exp(rng.uniform(np.log(c),np.log(a)))
        R=rand_rot(); Z=rng.uniform(8,20); xy=rng.uniform(-0.35,0.35,2)*Z
        T=np.array([xy[0],xy[1],Z]); Cp=outline_conic((a,b,c),R,T,F_PX)
        Cn=norm_sign(Cp)
        try: ctr,ax,V=ellipse_params(Cn)
        except Exception: continue
        if not np.all(np.isfinite(ax)) or np.any(np.abs(ctr)+ax.max()>W/2) or ax.min()<15: continue
        t=rng.uniform(0,2*np.pi,120); pts=ctr+(V@np.vstack([ax[0]*np.cos(t),ax[1]*np.sin(t)])).T
        pts+=rng.normal(scale=0.5,size=pts.shape)
        S=W; P=pts/S; f=F_PX/S                       # unit-size scaling
        Ct=norm_sign(np.diag([S,S,1])@Cn@np.diag([S,S,1]))  # true conic in scaled coords
        Ch=fit(P)
        boots=[]
        for _ in range(200):
            idx=rng.integers(0,len(P),len(P)); Cb=fit(P[idx])
            boots.append(np.linalg.norm(Cb-Ch,2))
        eps=np.quantile(boots,0.95); delta=max(f,1.0)**2*eps
        pos,_=kappa(Ch,f); mm,mp=pos
        if mm<=delta: klo,khi=1.0,np.inf
        else: klo=np.sqrt(max((mp-delta)/(mm+delta),1)); khi=np.sqrt((mp+delta)/(mm-delta))
        ktrue=np.sqrt(np.divide(*kappa(Ct,f)[0][::-1]))
        res[ratio].append((klo,khi,ktrue,np.linalg.norm(Ch-Ct,2)<=eps))
        n+=1
for ratio,L in res.items():
    L=np.array(L,dtype=float); klo,khi,kt,cov=L.T
    print(f"a/c={ratio}: true kappa in [lo,hi] {np.mean((kt>=klo-1e-12)&(kt<=khi+1e-12))*100:.1f}%  "
          f"||dC||<=eps {cov.mean()*100:.1f}%  "
          f"true certified consistent {np.mean(khi<=ratio)*100:.1f}%  true certified rejected {np.mean(klo>ratio)*100:.1f}%  "
          f"1.2 certified rejected {np.mean(klo>1.2)*100:.1f}%  1.5 certified rejected {np.mean(klo>1.5)*100:.1f}%  "
          f"median hi/lo {np.median(khi/klo):.3f}")
