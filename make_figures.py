"""Generate the six figures of the paper as vector PDFs."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, FancyArrowPatch
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'mathtext.fontset': 'cm'})
C1, C2, C3, GREY = '#1f77b4', '#d95f02', '#1b9e77', '#999999'

# ---------- Figure 1: the realizable set of sphere outlines, p known, f free
# An outline with long axis through p is described, up to scale, by u = b_E/a_E and v = d_E/a_E.
fig, ax = plt.subplots(figsize=(4.4, 3.2))
uu = np.linspace(0, 1, 400); arc = np.sqrt(1 - uu**2)
ax.fill_between(uu, arc, 2.2, color=C1, alpha=0.18, lw=0)
ax.plot(uu, arc, color=C2, lw=1.6, ls='--')
ax.plot([1, 1], [0, 2.2], color=C2, lw=1.6, ls='--')
ax.plot(1, 0, 'o', color=C1, ms=6, zorder=5, clip_on=False)
ax.text(0.25, 1.45, r'$\mathcal{R}$: sphere outlines', fontsize=9)
ax.text(0.08, 0.35, 'focus at $p$\n($f\\to0$): escape', color=C2, fontsize=8)
ax.text(1.03, 1.2, 'circles not\ncentered at $p$\n($f\\to\\infty$):\nescape', color=C2, fontsize=8)
ax.annotate('circle centered at $p$:\ncritical value (attained)', xy=(1, 0), xytext=(0.42, 0.08),
            fontsize=8, color=C1, arrowprops=dict(arrowstyle='->', color=C1))
ax.set_xlim(0, 1.45); ax.set_ylim(0, 2.2)
ax.set_xlabel(r'$u=b_E/a_E$'); ax.set_ylabel(r'$v=d_E/a_E$')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
fig.savefig('figures/fig1_boundary.pdf', bbox_inches='tight'); plt.close(fig)

# ---------- Figure 2: junctions, sectors and example principal points
from matplotlib.patches import Arc
def fit_interval(angles_deg, pdir_deg):
    th = np.radians(angles_deg); v = [np.array([np.cos(t), np.sin(t)]) for t in th]
    Nn = [np.array([-np.sin(t), np.cos(t)]) for t in th]
    p = np.array([np.cos(np.radians(pdir_deg)), np.sin(np.radians(pdir_deg))])
    h = [n @ (-p) for n in Nn]; pairs = [(0, 1), (1, 2), (0, 2)]
    ss = np.exp(np.linspace(np.log(1e-4), np.log(1e4), 400001)); P = np.ones_like(ss)
    for (a_, b_) in pairs: P *= ss*(v[a_] @ v[b_]) + h[a_]*h[b_]
    ok = ss[P < 0]
    return None if ok.size == 0 else (np.sqrt(ok.min()), np.sqrt(ok.max()))
def sector_arc(ax, a0, a1, r, label, col='0.35'):
    ax.add_patch(Arc((0, 0), 2*r, 2*r, theta1=a0, theta2=a1, color=col, lw=0.8))
    m = np.radians((a0 + a1)/2); ax.text((r + 0.13)*np.cos(m), (r + 0.13)*np.sin(m), label, fontsize=7.5,
                                       ha='center', va='center', color=col)
def draw_junction(ax, line_angles, half_edges, title):
    ax.set_aspect('equal'); ax.set_axis_off(); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    for k, a in enumerate(np.radians(line_angles)):
        ax.plot([-1.25*np.cos(a), 1.25*np.cos(a)], [-1.25*np.sin(a), 1.25*np.sin(a)], color='0.6', lw=0.7, ls='--')
    for a in np.radians(half_edges):
        ax.plot([0, 1.05*np.cos(a)], [0, 1.05*np.sin(a)], 'k', lw=2)
    ax.plot(0, 0, 'ko', ms=3); ax.text(0.05, -0.16, r'$x$', fontsize=8)
    ax.set_title(title, fontsize=9)
def mark_p(ax, ang, ok_text, good):
    q = 0.82*np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    if good:
        ax.plot(*q, 'o', color=C3, ms=5); ax.text(q[0], q[1] - 0.17, ok_text, color=C3, fontsize=7, ha='center', va='top')
    else:
        ax.plot(*q, 'x', color=C2, ms=7, mew=2); ax.text(q[0], q[1] + 0.14, ok_text, color=C2, fontsize=7, ha='center')

fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.3))
# (a) all double sectors acute: a fork drawn with half-edges at 0, 120, 240 degrees
draw_junction(axs[0], [0, 60, 120], [0, 120, 240], '(a) all double sectors acute')
sector_arc(axs[0], 0, 60, 0.42, r'$60^\circ$'); sector_arc(axs[0], 60, 120, 0.42, r'$60^\circ$'); sector_arc(axs[0], 120, 180, 0.42, r'$60^\circ$')
ia = fit_interval([0, 60, 120], 200)
mark_p(axs[0], 200, '', True)
axs[0].text(-0.88, -0.52, r'$p$: $f>%.2f\,|p-x|$' % ia[0], color=C3, fontsize=7, ha='center')
axs[0].text(0, -1.28, r'every $p$ works; $f$ bounded below only', fontsize=7.5, ha='center')
# (b) one obtuse double sector: an arrow with half-edges at 0, 30, 70 degrees
draw_junction(axs[1], [0, 30, 70], [0, 30, 70], '(b) one obtuse double sector')
for s0 in (0, 180):
    tt = np.radians(np.linspace(70 + s0, 180 + s0, 80))
    axs[1].fill(np.r_[0, 1.25*np.cos(tt)], np.r_[0, 1.25*np.sin(tt)], color=C2, alpha=0.15, lw=0)
sector_arc(axs[1], 0, 30, 0.42, r'$30^\circ$'); sector_arc(axs[1], 30, 70, 0.55, r'$40^\circ$'); sector_arc(axs[1], 70, 180, 0.35, r'$110^\circ$')
ib = fit_interval([0, 30, 70], 15)
mark_p(axs[1], 15, '', True)
axs[1].text(0.95, -0.12, r'$p$: $%.2f<f/|p-x|<%.2f$' % ib, color=C3, fontsize=7, ha='center', va='top')
mark_p(axs[1], 125, r'$p$: no $f$ fits', False)
axs[1].text(0, -1.28, r'$p$ in the shaded double sector is impossible', fontsize=7.5, ha='center')
fig.savefig('figures/fig2_junctions.pdf', bbox_inches='tight'); plt.close(fig)

# ---------- Figure 3: W_J for junction (b)
ang = np.radians([0, 30, 70]); v = [np.array([np.cos(a), np.sin(a)]) for a in ang]
N = [np.array([-np.sin(a), np.cos(a)]) for a in ang]
g = np.linspace(-3, 3, 600); WX, WY = np.meshgrid(g, g); prod = np.ones_like(WX)
for i, j in [(0, 1), (1, 2), (0, 2)]:
    prod *= v[i] @ v[j] + (N[i][0]*WX + N[i][1]*WY)*(N[j][0]*WX + N[j][1]*WY)
fig, ax = plt.subplots(figsize=(3.4, 3.4)); ax.set_aspect('equal')
ax.contourf(WX, WY, (prod < 0).astype(float), levels=[0.5, 1.5], colors=[C1], alpha=0.25)
ax.contour(WX, WY, prod, levels=[0], colors=[C1], linewidths=0.8)
d = np.array([np.cos(np.radians(32)), np.sin(np.radians(32))])
ax.plot([0, 3*d[0]], [0, 3*d[1]], color=C2, lw=1.2); ax.plot(0, 0, 'ko', ms=3)
ax.text(0.08, -0.3, r'$0$ ($f\to\infty$)', fontsize=8); ax.text(0.9, -0.9, r'ray $w=(x-p)/f$', color=C2, fontsize=8)
ax.set_xlabel(r'$w_1$'); ax.set_ylabel(r'$w_2$'); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
fig.savefig('figures/fig3_WJ.pdf', bbox_inches='tight'); plt.close(fig)

# ---------- Figure 4: 3:2:1 ellipsoid, focal hyperbola and middle axis (section y = 0)
a, b, c = 3., 2., 1.
fig, ax = plt.subplots(figsize=(4.6, 3.0)); ax.set_aspect('equal')
ax.add_patch(Ellipse((0, 0), 2*a, 2*c, fill=True, fc='0.9', ec='k', lw=0.8))
s = np.linspace(-1.6, 1.6, 300); A2, C2_ = a*a - b*b, b*b - c*c
for sg in (1, -1):
    x = sg*np.sqrt(A2)*np.cosh(s); z = np.sqrt(C2_)*np.sinh(s)
    inside = x**2/a**2 + z**2/c**2 < 1
    ax.plot(np.where(inside, np.nan, x), z, color=C1, lw=1.5)
    ax.plot(np.where(inside, x, np.nan), z, color=C1, lw=1.0, ls=':')
ax.plot(0, 0, 'k+', ms=8)
ax.text(-0.9, -0.35, r'middle axis $\perp$ plane', fontsize=7)
ax.text(-2.4, 1.5, r'focal hyperbola: ratio $1$', color=C1, fontsize=8)
ax.text(-4.9, -2.6, r'section $O_2=0$; from the middle axis the ratio is $a^2/c^2$', fontsize=7)
ax.set_xlim(-5.2, 5.2); ax.set_ylim(-3, 3); ax.set_xlabel(r'$O_1$'); ax.set_ylabel(r'$O_3$')
fig.savefig('figures/fig4_ellipsoid.pdf', bbox_inches='tight'); plt.close(fig)

# ---------- Figure 5: kappa(f) for the Example ellipse
def conic(cx, cy, sa, sb, th):
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); Ae = R@np.diag([1/sa**2, 1/sb**2])@R.T
    cc = np.array([cx, cy]); return np.block([[Ae, -(Ae@cc)[:, None]], [-(Ae@cc)[None, :], np.array([[cc@Ae@cc - 1]])]])
Cm = conic(320, 140, 60, 38, 0.5)
def kappa(f):
    K = np.diag([f, f, 1.]); ev = np.linalg.eigvalsh(K@Cm@K); ev = ev if (ev > 0).sum() == 2 else -ev
    pos = np.sort(ev[ev > 0]); return np.sqrt(pos[1]/pos[0])
fs = np.exp(np.linspace(np.log(50), np.log(1e5), 2000)); ks = np.array([kappa(f) for f in fs])
i = ks.argmin(); print('kappa_min %.3f at f=%.0f, ortho %.3f' % (ks[i], fs[i], ks[-1]))
ins = fs[ks <= 1.5]; print('I(1.5) = [%.0f, %.0f]' % (ins.min(), ins.max())); print('I(2) from %.0f' % fs[ks <= 2].min())
fig, ax = plt.subplots(figsize=(4.6, 2.8)); ax.semilogx(fs, ks, 'k', lw=1.2)
for lev, col in [(1.5, C1), (2.0, C3)]: ax.axhline(lev, color=col, lw=0.8, ls='--'); ax.text(60, lev + 0.04, rf'$a/c={lev:g}$', color=col, fontsize=8)
ax.axhline(ks[-1], color=GREY, lw=0.6, ls=':'); ax.plot(fs[i], ks[i], 'o', color=C2, ms=4)
ax.set_xlabel(r'$f$ (px)'); ax.set_ylabel(r'$\kappa(f)$'); ax.set_ylim(1, 3)
fig.savefig('figures/fig5_kappa.pdf', bbox_inches='tight'); plt.close(fig)

# ---------- Figure 6: confocal hyperbolas
A, B = 4., 1.
def xa(mu, z): return np.sqrt(A + mu)*np.sqrt(1 + z*z/(-mu))
zs = np.linspace(0, 4, 400)
fig, ax = plt.subplots(figsize=(5.0, 2.8)); ax.set_aspect('equal')
for sg in (1, -1):
    xin, xout = sg*xa(-2.25, zs), sg*xa(-1/2.25, zs)
    ax.fill_betweenx(zs, xin, xout, color=C2, alpha=0.15, lw=0)
    ax.plot(xin, zs, color=C2, lw=1); ax.plot(xout, zs, color=C2, lw=1)
    ax.plot(sg*xa(-0.25, zs), zs, color=C3, lw=1, ls='--'); ax.plot(sg*xa(-1, zs), zs, color=C1, lw=1.6)
ax.plot([-2, 2], [0, 0], 'k', lw=3); ax.plot([-np.sqrt(3), np.sqrt(3)], [0, 0], 'o', color=C1, ms=4)
ax.plot([], [], color=C1, lw=1.6, label=r'$\kappa=1$ (focal hyperbola)'); ax.plot([], [], color=C2, lw=1, label=r'$\kappa=1.5$')
ax.plot([], [], color=C3, lw=1, ls='--', label=r'$\kappa=2$'); ax.legend(fontsize=7, loc='upper center', ncol=3, frameon=False)
ax.set_xlim(-7, 7); ax.set_ylim(0, 4.6); ax.set_xlabel('position along the long axis of $E$'); ax.set_ylabel('$f$')
fig.savefig('figures/fig6_confocal.pdf', bbox_inches='tight'); plt.close(fig)

# ---------- Figure 7: two loci meeting (sphere outline and rectangle), vertical plane through the sphere axis
xc, aE, bE = 4.0, 2.0, 1.2; eE = np.sqrt(aE**2 - bE**2)
cx, rho = 0.0, 2.5            # section of the rectangle's Thales sphere by this plane
fig, ax = plt.subplots(figsize=(5.4, 2.9)); ax.set_aspect('equal')
s_ = np.linspace(0, 2.2, 300)
for sg in (-1, 1):
    hx = xc + sg*eE*np.cosh(s_); hz = bE*np.sinh(s_)
    ax.plot(hx, hz, color=C1, lw=1.6)
tt = np.linspace(0, np.pi, 300); ax.plot(cx + rho*np.cos(tt), rho*np.sin(tt), color=C2, lw=1.6)
# intersection of the left branch with the circle
from scipy.optimize import brentq
F = lambda s0: (xc - eE*np.cosh(s0) - cx)**2 + (bE*np.sinh(s0))**2 - rho**2
s_star = brentq(F, 0.01, 2.0); Ox, Oz = xc - eE*np.cosh(s_star), bE*np.sinh(s_star)
ax.plot(Ox, Oz, 'ko', ms=5); ax.text(Ox + 0.12, Oz + 0.12, r'$O=(p,f)$', fontsize=8)
ax.plot([Ox, Ox], [0, Oz], 'k:', lw=0.8); ax.plot(Ox, 0, 'k|', ms=8); ax.text(Ox - 0.08, -0.38, r'$p$', fontsize=8)
ax.plot([xc - aE, xc + aE], [0, 0], color=C1, lw=4, alpha=0.5)
ax.text(xc - 0.9, -0.42, 'sphere outline (long axis)', fontsize=7, color=C1)
ax.text(5.5, 2.3, 'focal hyperbola:\nsphere locus', fontsize=7.5, color=C1)
ax.text(-2.9, 2.35, 'Thales sphere:\nrectangle locus', fontsize=7.5, color=C2)
ax.axhline(0, color='0.5', lw=0.6)
ax.set_xlim(-3.2, 8.2); ax.set_ylim(-0.6, 3.0); ax.set_xlabel('position along the sphere outline\'s long axis'); ax.set_ylabel('$f$')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
fig.savefig('figures/fig7_loci.pdf', bbox_inches='tight'); plt.close(fig)
print('camera at', round(Ox, 3), round(Oz, 3))
