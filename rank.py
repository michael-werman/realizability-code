"""Numerical generic rank of the image map for the main families (p known, f free)."""
import numpy as np
rng=np.random.default_rng(3)
def rotv(w):
    th=np.linalg.norm(w); k=w/th if th>0 else np.zeros(3); Kx=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(th)*Kx+(1-np.cos(th))*Kx@Kx
def jac_rank(F,x0,h=None):
    y0=F(x0); hs=np.maximum(np.abs(x0)*1e-6,1e-7); J=np.array([(F(x0+hh*e)-F(x0-hh*e))/(2*hh) for hh,e in zip(hs,np.eye(len(x0)))]).T
    s=np.linalg.svd(J,compute_uv=False); return int((s>s[0]*1e-11).sum()), len(x0), len(y0)
def conic_vec(C):
    C=C/np.linalg.norm(C); C=C*np.sign(C[0,0]) ; return np.array([C[0,0],C[0,1],C[1,1],C[0,2],C[1,2],C[2,2]])
def conic_feat(C):  # 5 numbers: normalize by C[2,2]... use projective normalization
    v=conic_vec(C); return v[:5]/v[5]
# corner: params f, rotation(3), ray direction(2) -> junction: vertex(2), three line angles
def corner(x):
    f=x[0]; R=rotv(x[1:4]); d=np.array([x[4],x[5],1.0]); X0=5*d
    xs=lambda X: f*X[:2]/X[2]
    v=xs(X0); ang=[]
    for k in range(3):
        X1=X0+0.1*R[:,k]; u=xs(X1)-v; ang.append(np.arctan2(u[1],u[0]))
    return np.r_[v,ang]
# sphere: center(3), radius(1), f -> conic (5)
def sphere(x):
    c=x[:3]; r=x[3]; f=x[4]; M=np.outer(c,c)-(c@c-r*r)*np.eye(3); Ki=np.diag([1/f,1/f,1]); return conic_feat(Ki.T@M@Ki)
# ellipsoid of known shape (3:2:1): center(3), rotation(3), scale(1), f -> conic(5)
def ellipsoid(x):
    T=x[:3]; R=rotv(x[3:6]); s=x[6]; f=x[7]; A=R@np.diag(1/(s*np.array([3,2,1.]))**2)@R.T
    S=T@A@T-1; u=A@T; M=S*A-np.outer(u,u); Ki=np.diag([1/f,1/f,1]); return conic_feat(Ki.T@M@Ki)
# rectangle: center(3), rotation(3), width w, aspect a, f -> 8 image coordinates
def rect(x):
    T=x[:3]; R=rotv(x[3:6]); w=x[6]; a=x[7]; f=x[8]
    P=[T+R@np.array([sx*w,sy*w*a,0]) for sx,sy in ((-1,-1),(1,-1),(1,1),(-1,1))]
    return np.concatenate([f*p[:2]/p[2] for p in P])
cases={'right-angle corner (n=6, dim F=5)':(corner,lambda: np.r_[rng.uniform(500,1500),rng.normal(size=3)*0.6,rng.normal(size=2)*0.2]),
       'sphere (n=4, dim F=5)':(sphere,lambda: np.r_[rng.normal(size=2)*1.5,rng.uniform(8,12),rng.uniform(0.5,1),rng.uniform(500,1500)]),
       'ellipsoid 3:2:1 (n=7, dim F=5)':(ellipsoid,lambda: np.r_[rng.normal(size=2)*1.5,rng.uniform(15,20),rng.normal(size=3),rng.uniform(0.5,1),rng.uniform(500,1500)]),
       'rectangle, any aspect (n=8, dim F=8)':(rect,lambda: np.r_[rng.normal(size=2),rng.uniform(8,12),rng.normal(size=3)*0.5,rng.uniform(0.5,1),rng.uniform(0.5,2),rng.uniform(500,1500)])}
for name,(F,gen) in cases.items():
    ranks=[jac_rank(F,gen()) for _ in range(20)]
    print(name, 'raw params',ranks[0][1],'features',ranks[0][2],'ranks',sorted(set(r[0] for r in ranks)))
