"""Composite-image demo: a box and a sphere from one camera, and a sphere pasted in from another.
Tests: box -> (p, f) from the vanishing points (orthocenter); sphere -> long axis through p and focal formula.
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False, 'mathtext.fontset': 'stix', 'font.family': 'STIXGeneral'})
from matplotlib.patches import Polygon, Ellipse
rng = np.random.default_rng(7)
W, H = 1000, 750
def Kmat(f, p): return np.array([[f, 0, p[0]], [0, f, p[1]], [0, 0, 1.0]])
def proj(K, X): x = (K @ X.T).T; return x[:, :2]/x[:, 2:3]
def rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]); return Rz @ Rx @ Ry
def sphere_conic(K, c, r):
    # cone of rays tangent to the sphere: d^T (r^2 I - (|c|^2 I - c c^T)) d ... use (c.d)^2 = (|c|^2 - r^2)|d|^2
    M = np.outer(c, c) - (c @ c - r*r)*np.eye(3)       # d^T M d = 0 on the tangent cone
    Ki = np.linalg.inv(K); C = Ki.T @ M @ Ki
    return -C/np.linalg.norm(C) if np.linalg.eigvalsh(C[:2, :2]).min() < 0 else C/np.linalg.norm(C)
def ellipse_params(C):
    A = C[:2, :2]; b = C[:2, 2]; c0 = C[2, 2]; ctr = -np.linalg.solve(A, b); k = -(c0 + b @ ctr)
    ev, V = np.linalg.eigh(A); ax = np.sqrt(k/ev); return ctr, ax, V   # ax[0] >= ax[1] (ev ascending)
def sample_ellipse(C, n=150, sigma=0.5):
    ctr, ax, V = ellipse_params(C); t = rng.uniform(0, 2*np.pi, n)
    pts = ctr + (V @ np.vstack([ax[0]*np.cos(t), ax[1]*np.sin(t)])).T
    return pts + rng.normal(scale=sigma, size=pts.shape)
def fit_conic(pts):
    s = 1000.0; x, y = pts[:, 0]/s, pts[:, 1]/s
    D = np.column_stack([x*x, x*y, y*y, x, y, np.ones_like(x)]); a, b, c, d, e, g = np.linalg.svd(D)[2][-1]
    C = np.array([[a, b/2, d/2], [b/2, c, e/2], [d/2, e/2, g]]); S = np.diag([1/s, 1/s, 1]); C = S @ C @ S
    return -C/np.linalg.norm(C) if np.linalg.eigvalsh(C[:2, :2]).min() < 0 else C/np.linalg.norm(C)
def line_through(pts):            # total least squares line, returned as homogeneous l with l.(x,y,1)=0
    m = pts.mean(0); u = np.linalg.svd(pts - m)[2][0]; n = np.array([-u[1], u[0]]); return np.array([n[0], n[1], -n @ m])
def vp(lines):                     # least-squares intersection of lines
    A = np.array([l[:2]/np.linalg.norm(l[:2]) for l in lines]); b = np.array([-l[2]/np.linalg.norm(l[:2]) for l in lines])
    return np.linalg.lstsq(A, b, rcond=None)[0]
def orthocenter(a, b, c):
    A = np.array([b - c, a - c]); rhs = np.array([(b - c) @ a, (a - c) @ b]); return np.linalg.solve(A, rhs)

# ---- scene, camera A
fA, pA = 900.0, np.array([500.0, 375.0]); KA = Kmat(fA, pA)
Rb = rot(np.radians(38), np.radians(28), np.radians(4)); cb = np.array([-1.3, 0.4, 9.0]); half = 1.1
corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
Xbox = cb + (Rb @ (half*corners).T).T; xbox = proj(KA, Xbox)
edges = [(i, j) for i in range(8) for j in range(i+1, 8) if np.sum(corners[i] != corners[j]) == 1]
def edge_axis(i, j): return int(np.argmax(corners[i] != corners[j]))
sphere_true = (np.array([2.0, 0.9, 10.5]), 0.9)               # photographed with camera A
C_true = sphere_conic(KA, *sphere_true)
# ---- pasted sphere: photographed by camera B (shorter focal length), then translated in the image
fB, pB = 420.0, np.array([500.0, 375.0]); KB = Kmat(fB, pB)
C_B = sphere_conic(KB, np.array([1.6, 1.0, 6.0]), 0.75)
ctrB, _, _ = ellipse_params(C_B); rdir = (ctrB - pB)/np.linalg.norm(ctrB - pB); target = pA + 400.0*rdir; T = np.array([[1, 0, target[0]-ctrB[0]], [0, 1, target[1]-ctrB[1]], [0, 0, 1]])
Ti = np.linalg.inv(T); C_fake = Ti.T @ C_B @ Ti; C_fake /= np.linalg.norm(C_fake)

def box_estimate(sig=0.5):
    groups = {0: [], 1: [], 2: []}
    for i, j in edges:
        t = np.linspace(0, 1, 30)[:, None]; pts = xbox[i]*(1 - t) + xbox[j]*t + rng.normal(scale=sig, size=(30, 2))
        groups[edge_axis(i, j)].append(line_through(pts))
    v = [vp(groups[k]) for k in range(3)]; p = orthocenter(*v)
    f2 = np.mean([-(v[a]-p) @ (v[b]-p) for a, b in ((0, 1), (1, 2), (0, 2))]); return p, np.sqrt(f2), v
def sphere_test(C, p):
    ctr, ax, V = ellipse_params(C); aE, bE = ax[0], ax[1]; u = V[:, 0]            # long-axis direction
    n = np.array([-u[1], u[0]]); off = abs(n @ (p - ctr))                         # distance of p from the long-axis line
    dE = abs(u @ (p - ctr)); eE = np.sqrt(aE**2 - bE**2)
    f = np.sqrt(bE**2*(dE**2 - aE**2 + bE**2)/(aE**2 - bE**2)) if dE > eE else np.nan
    return off, f, ctr, aE, bE, u
# ---- Monte Carlo over measurement noise
rows = {'real': [], 'pasted': []}
for _ in range(300):
    p, f, v = box_estimate()
    for name, C in (('real', C_true), ('pasted', C_fake)):
        off, fs, *_ = sphere_test(fit_conic(sample_ellipse(C)), p)
        rows[name].append((p[0], p[1], f, off, fs))
for name, R in rows.items():
    R = np.array(R)
    print(name, 'box p = (%.1f, %.1f) +- (%.1f, %.1f)  f_box = %.1f +- %.1f' % (R[:, 0].mean(), R[:, 1].mean(), R[:, 0].std(), R[:, 1].std(), R[:, 2].mean(), R[:, 2].std()),
          ' | axis misses p by %.1f +- %.1f px, f_sphere = %.1f +- %.1f' % (R[:, 3].mean(), R[:, 3].std(), np.nanmean(R[:, 4]), np.nanstd(R[:, 4])))
np.save('rows.npy', {k: np.array(v) for k, v in rows.items()}, allow_pickle=True)

# ---- render: (a) the composite, (b) the image test, (c) the loci
p_hat, f_hat, v_hat = box_estimate()
R = {k: np.array(v) for k, v in rows.items()}
COL = {'real': '#1f5fb4', 'pasted': '#c0392b'}
def draw_scene(ax, annotate):
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Rectangle((0, 0), W, H, color='#e9e4da'))
    ax.add_patch(plt.Rectangle((0, 455), W, H - 455, color='#d3cbbd'))
    faces = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    Xc = Xbox.mean(0); light = np.array([0.5, 0.75, 0.45]); light /= np.linalg.norm(light)
    for fc in faces:
        P = Xbox[list(fc)]; nrm = np.cross(P[1] - P[0], P[2] - P[0]); nrm /= np.linalg.norm(nrm)
        if nrm @ (P.mean(0) - Xc) < 0: nrm = -nrm
        if nrm @ P.mean(0) < 0:
            shade = 0.45 + 0.5*max(0, -nrm @ light)
            ax.add_patch(Polygon(xbox[list(fc)], closed=True, fc=(0.55*shade, 0.38*shade, 0.25*shade), ec='#3a2a1c', lw=1.2))
    for C, col in ((C_true, (0.25, 0.42, 0.72)), (C_fake, (0.72, 0.28, 0.25))):
        ctr, axs_, V = ellipse_params(C); ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
        for k, sc in enumerate(np.linspace(1.0, 0.15, 14)):
            g = 0.55 + 0.45*k/13; off = (1 - sc)*np.array([-0.35*axs_[1], -0.35*axs_[1]])
            ax.add_patch(Ellipse(ctr + off, 2*axs_[0]*sc, 2*axs_[1]*sc, angle=ang, color=tuple(min(1, c*g + 0.25*k/13) for c in col), lw=0))
    if annotate:
        ax.plot(*p_hat, 'k+', ms=14, mew=2.5)
        ax.annotate(r'$p$ from the box', p_hat, xytext=(p_hat[0] - 380, p_hat[1] - 120), fontsize=12, arrowprops=dict(arrowstyle='->', lw=0.8))
        for name, C in (('real', C_true), ('pasted', C_fake)):
            off, fs, ctr, aE, bE, u = sphere_test(C, p_hat); L = 1600
            ax.plot([ctr[0] - L*u[0], ctr[0] + L*u[0]], [ctr[1] - L*u[1], ctr[1] + L*u[1]], color=COL[name], lw=1.2, ls='--')
        ax.text(20, 730, 'both long axes pass through $p$', fontsize=12)
fig = plt.figure(figsize=(15, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95])
a0, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
draw_scene(a0, False); a0.set_title('(a) composite: box and two balls', fontsize=13)
draw_scene(a1, True); a1.set_title('(b) the axis test passes for both balls', fontsize=13)
# (c) loci: each ball's focal hyperbola in its axial plane, with t = signed distance from p along the axis
for name, C in (('real', C_true), ('pasted', C_fake)):
    off, fs, ctr, aE, bE, u = sphere_test(C, p_hat); eE = np.sqrt(aE**2 - bE**2); d = abs(u @ (p_hat - ctr))
    x = np.linspace(eE, d + 400, 600); t = d - x                       # t = 0 at p, t > 0 towards the ball
    z = bE*np.sqrt(np.maximum(x**2/eE**2 - 1, 0))
    lab = 'real ball: $f=%.0f$ at $p$' % np.nanmean(R['real'][:, 4]) if name == 'real' else 'pasted ball: $f=%.0f$ at $p$' % np.nanmean(R['pasted'][:, 4])
    a2.plot(t, z, color=COL[name], lw=2, label=lab)
for name in ('real', 'pasted'):
    fs_ = R[name][:, 4]; a2.errorbar([0], [np.nanmean(fs_)], yerr=[2*np.nanstd(fs_)], fmt='s', color=COL[name], ms=6, capsize=4, mfc='white')
fb = R['real'][:, 2]
a2.errorbar([0], [fb.mean()], yerr=[2*fb.std()], fmt='ko', ms=7, capsize=4, label=r'box: $O=(p,f)$, $f=%.0f$' % fb.mean())
a2.axvline(0, color='0.6', lw=0.8, ls=':')
a2.set_xlim(-400, 260); a2.set_ylim(0, 1700); a2.set_xlabel('position along the ball\'s long axis, from $p$ (px)', fontsize=11)
a2.set_ylabel('$f$ (px)', fontsize=11); a2.legend(fontsize=10, loc='upper left', frameon=False)
a2.set_title('(c) the loci: only the real ball meets the box', fontsize=13)
for sp in ['top', 'right']: a2.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('composite_demo.png', dpi=150); fig.savefig('composite_demo.pdf')
print('p_hat', p_hat.round(1), 'f_hat', round(f_hat, 1))
