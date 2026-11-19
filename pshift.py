"""Sensitivity of the ellipse test to an error in the principal point (point estimate, f known)."""
import numpy as np
exec(open('certified.py').read().split('res={r:[]')[0])   # reuse helpers
rng = np.random.default_rng(11)
def shift(C, d):   # conic in coordinates centered at p; move the assumed p by d
    T = np.array([[1, 0, d[0]], [0, 1, d[1]], [0, 0, 1.0]]); return T.T @ C @ T
out = {}
for ratio in (1.5, 2.0, 3.0):
    rows = []; n = 0
    while n < 200:
        a, c = ratio, 1.0; b = np.exp(rng.uniform(np.log(c), np.log(a)))
        R = rand_rot(); Z = rng.uniform(8, 20); xy = rng.uniform(-0.35, 0.35, 2)*Z
        Cn = norm_sign(outline_conic((a, b, c), R, np.array([xy[0], xy[1], Z]), F_PX))
        try: ctr, ax, V = ellipse_params(Cn)
        except Exception: continue
        if not np.all(np.isfinite(ax)) or np.any(np.abs(ctr) + ax.max() > W/2) or ax.min() < 15: continue
        t = rng.uniform(0, 2*np.pi, 120); pts = ctr + (V @ np.vstack([ax[0]*np.cos(t), ax[1]*np.sin(t)])).T
        pts += rng.normal(scale=0.5, size=pts.shape)
        r = []
        for dp in (0, 10, 25, 50):
            ang = rng.uniform(0, 2*np.pi); d = dp*np.array([np.cos(ang), np.sin(ang)])
            P = (pts - d)/W; Ch = fit(P); pos, _ = kappa(Ch, F_PX/W); k = np.sqrt(pos[1]/pos[0])
            r.append((k <= ratio + 1e-12, k > 1.2))
        rows.append(r); n += 1
    out[ratio] = np.array(rows, float)
for ratio, A in out.items():
    print(ratio, 'accept true:', [f'{v*100:.1f}%' for v in A[:, :, 0].mean(0)], ' reject 1.2:', [f'{v*100:.1f}%' for v in A[:, :, 1].mean(0)])
