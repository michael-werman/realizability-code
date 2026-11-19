"""Symbolic and numerical checks of the identities used in
'Which Images Can a Family of 3D Shapes Produce?'.  Run: python verify_identities.py"""
import sympy as sp, numpy as np
ok = lambda name, cond: print(("PASS " if cond else "FAIL ") + name)

# 1. Corner: g_ij = (f^2 c_ij + h_i h_j)|r|^2  (Section 5)
f, x1, x2, t1, t2 = sp.symbols('f x1 x2 t1 t2', real=True)
v = lambda t: sp.Matrix([sp.cos(t), sp.sin(t)])
N = lambda t: sp.Matrix([-sp.sin(t), sp.cos(t)])
x = sp.Matrix([x1, x2])
def n_vec(t):  # K^T l with l = (N, -N.x), K = diag(f,f,1)
    return sp.Matrix([f*N(t)[0], f*N(t)[1], -(N(t).dot(x))])
r = sp.Matrix([x1/f, x2/f, 1])
ni, nj = n_vec(t1), n_vec(t2)
g = (ni.cross(r)).dot(nj.cross(r))
rhs = (f**2*v(t1).dot(v(t2)) + N(t1).dot(x)*N(t2).dot(x)) * (r.dot(r))
ok("corner identity g_ij", sp.simplify(sp.expand_trig(g*f**2 - rhs*f**2)) == 0)

# 2. Lemma 12: secular equation of M = sA - w w^T equals the confocal equation
a2, b2, c2, O1, O2, O3, lam = sp.symbols('a2 b2 c2 O1 O2 O3 lam', positive=True)
A = sp.diag(1/a2, 1/b2, 1/c2); O = sp.Matrix([O1, O2, O3])
s = (O.T*A*O)[0] - 1; w = A*O
M = s*A - w*w.T
nu = -s/lam
# numeric root check
rng = np.random.default_rng(0); good = True
for _ in range(200):
    av = sorted(rng.uniform(0.5, 3, 3))[::-1]; Ov = rng.normal(0, 3, 3)
    Am = np.diag(1/np.array(av)**2); sv = Ov@Am@Ov - 1
    if sv <= 0: continue
    Mv = sv*Am - np.outer(Am@Ov, Am@Ov); ev = np.sort(np.linalg.eigvalsh(Mv))
    P = np.polynomial.polynomial
    A2 = np.array(av)**2
    den = P.polymul(P.polymul([A2[0],1],[A2[1],1]),[A2[2],1])
    num = sum(Ov[i]**2*P.polymul(*[[A2[j],1] for j in range(3) if j!=i]) for i in range(3))
    lams = P.polyroots(P.polysub(den, num)).real
    good &= np.allclose(np.sort(-sv/lams), ev, rtol=1e-8, atol=1e-10)
ok("Lemma 12 eigenvalues = -s/lambda_i (200 random viewpoints)", good)

# 3. Inverse confocal formula (Theorem 13, Step 2): partial fractions
l1, l2, l3 = sp.symbols('l1 l2 l3')
av = [a2, b2, c2]; ls = [l1, l2, l3]
Oi2 = [sp.prod([av[i]+lj for lj in ls]) / sp.prod([av[i]-av[k] for k in range(3) if k != i]) for i in range(3)]
pf = sum(Oi2[i]/(av[i]+lam) for i in range(3)) - 1 + sp.prod([lam-lj for lj in ls])/sp.prod([ai+lam for ai in av])
ok("inverse confocal formula (partial fractions)", sp.simplify(pf) == 0)

# 4. Proposition 16: values of R_k
u, k, al1, al2, c0, p1, p2 = sp.symbols('u k alpha1 alpha2 c0 p1 p2')
P_ = sp.Function('P'); q = lambda z: (z-al1)*(z-al2)
Rk = lambda z, Pz, Pkz: (k*Pz*q(k*z) - Pkz*q(z))/(1-k)
detC = sp.Symbol('detC')
ok("R_k(0) = -alpha1 alpha2 detC", sp.simplify(Rk(0, detC, detC) + al1*al2*detC) == 0)
Pa1 = sp.Symbol('Pa1'); Pa2 = sp.Symbol('Pa2'); X = sp.Symbol('X')
ok("R_k(alpha1/k) = P(a1) q(a1/k)/(k-1)", sp.simplify(Rk(al1/k, X, Pa1) - Pa1*q(al1/k)/(k-1)) == 0)
ok("R_k(alpha1) = -k a1 P(a1)(k a1 - a2)", sp.simplify(Rk(al1, Pa1, X) + k*al1*Pa1*(k*al1-al2)) == 0)
ok("R_k(alpha2) = k P(a2) q(k a2)/(1-k)", sp.simplify(Rk(al2, Pa2, X) - k*Pa2*q(k*al2)/(1-k)) == 0)
# leading coefficient with P(z) = c0 z^2 + p1 z + p2
Pz = lambda z: c0*z**2 + p1*z + p2
lead = sp.Poly(sp.expand((k*Pz(u)*q(k*u) - Pz(k*u)*q(u))/(1-k)), u).LC()
ok("leading coefficient of R_k is -c0 k^2", sp.simplify(lead + c0*k**2) == 0)

# 5. Sphere focal formula (Section 8)
fa, al, th = sp.symbols('f alpha theta', positive=True)
D = sp.cos(al)**2 - sp.sin(th)**2
a_ = fa*sp.sin(al)*sp.cos(al)/D; b_ = fa*sp.sin(al)/sp.sqrt(D); c_ = fa*sp.sin(th)*sp.cos(th)/D
ok("sphere: f^2(a^2-b^2) = b^2(c^2-a^2+b^2)", sp.simplify(sp.expand_trig(fa**2*(a_**2-b_**2) - b_**2*(c_**2-a_**2+b_**2))) == 0)

# 6. Proposition 20(4): support-function difference
T, ph = sp.symbols('theta phi')
ok("cos^2(T)-cos^2(T-phi) = -sin(2T-phi) sin(phi)",
   sp.simplify(sp.expand_trig(sp.cos(T)**2 - sp.cos(T-ph)**2 + sp.sin(2*T-ph)*sp.sin(ph))) == 0)

# 7. Theorem 7 sign table (numerical)
rng = np.random.default_rng(1); good = True
for _ in range(20000):
    th3 = sorted(rng.uniform(0, np.pi, 3)); ang = np.array(th3)
    xv = rng.normal(0, 1, 2); pv = rng.normal(0, 1, 2)
    vs = [np.array([np.cos(t), np.sin(t)]) for t in ang]
    Ns = [np.array([-np.sin(t), np.cos(t)]) for t in ang]
    h = [Ni@(xv-pv) for Ni in Ns]
    pairs = [(0,1),(1,2),(0,2)]
    c = [vs[i]@vs[j] for i,j in pairs]; hh = [h[i]*h[j] for i,j in pairs]
    roots = [-hh[m]/c[m] for m in range(3)]
    npos = sum(rr > 0 for rr in roots)
    secs = [ang[1]-ang[0], ang[2]-ang[1], np.pi-(ang[2]-ang[0])]
    obtuse = max(secs) > np.pi/2
    good &= (npos == 1) if not obtuse else (npos in (0, 2))
ok("Theorem 7: one positive root (all acute) or 0/2 (one obtuse), 20000 junctions", good)

# 8. Certified ellipse test (Proposition in Section 6): Weyl bound
rng = np.random.default_rng(2); good = True
for _ in range(5000):
    a, b = sorted(rng.uniform(20, 300, 2))[::-1]; th = rng.uniform(0, np.pi); cx, cy = rng.normal(0, 200, 2)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); Ae = R@np.diag([1/a**2, 1/b**2])@R.T
    cc = np.array([cx, cy]); C = np.block([[Ae, -(Ae@cc)[:, None]], [-(Ae@cc)[None, :], np.array([[cc@Ae@cc-1]])]])
    C /= np.linalg.norm(C)
    E = rng.normal(0, 1, (3, 3)); E = (E+E.T)/2; eps = 10**rng.uniform(-6, -3); E *= eps/np.linalg.norm(E, 2)
    fv = np.exp(rng.uniform(np.log(100), np.log(5000))); K = np.diag([fv, fv, 1.])
    sg = 1 if (np.linalg.eigvalsh(K@C@K) > 0).sum() == 2 else -1
    def mus(Cm):
        ev = np.sort(np.linalg.eigvalsh(sg*K@Cm@K))
        return ev[1:]  # the two largest (positive) eigenvalues
    mt = mus(C); mh = mus(C+E); d = max(fv, 1)**2*eps
    if mh[0] <= d: continue
    lo = (mh[1]-d)/(mh[0]+d); hi = (mh[1]+d)/(mh[0]-d)
    good &= lo <= mt[1]/mt[0] <= hi
ok("certified ellipse test: true ratio inside the Weyl interval (5000 trials)", good)

# 9. Equivalence with Theorem 1 of Gaudilliere, Simon and Berger (2023): scan their parameter m
rng = np.random.default_rng(5); agree = 0; n = 0; near = 0
for _ in range(600):
    a, b, c = sorted(rng.uniform(0.5, 3, 3))[::-1]
    if a-b < 0.05 or b-c < 0.05: continue
    mu0, mm = np.exp(rng.uniform(-2, 2, 2)); rho = np.exp(rng.uniform(0, np.log(4*(a/c)**2)))
    Bp = np.diag([-mu0, mm*rho, mm])*rng.choice([-1, 1])
    lA = np.array([1/a**2, 1/b**2, 1/c**2]); MA = np.vstack([np.ones(3), lA, lA**2])
    d = np.cbrt(np.prod(lA)/np.linalg.det(Bp))
    ms = -np.exp(np.linspace(np.log(1e-4), np.log(1e4), 20001))
    V = np.vstack([np.sum(1/lA)-np.trace(np.linalg.inv(Bp))*ms/d, 1-ms**3, np.trace(Bp)*d*ms**2-lA.sum()*ms**3])
    D2 = np.linalg.solve(MA, V)
    theirs = np.any(np.all(D2 >= 0, axis=0)); ours = rho <= (a/c)**2
    n += 1; agree += theirs == ours; near += (theirs != ours) and abs(rho/(a/c)**2-1) < 1e-3
ok(f"P1E Theorem 1 agrees with the ratio test ({agree}/{n}; disagreements only within 1e-3 of the bound: {n-agree == near})", n-agree == near)
