import numpy as np

# f(x) = x1^4 + 2x1^3 + 24x1^2 + x2^4 + 12x2^2
f    = lambda x: x[0]**4 + 2*x[0]**3 + 24*x[0]**2 + x[1]**4 + 12*x[1]**2
grad = lambda x: np.array([4*x[0]**3 + 6*x[0]**2 + 48*x[0], 4*x[1]**3 + 24*x[1]])
hess = lambda x: np.diag([12*x[0]**2 + 12*x[0] + 48, 12*x[1]**2 + 24])

def subproblema(g, H, gam):
    """min g.d + 1/2 d.Hd  s.a. ||d||<=gam  (exacto, H simetrica).
    Newton si H>0 y ||H^-1 g||<=gam; si no, (H+lam I)d=-g con ||d|| = gam por biseccion."""
    w, Q = np.linalg.eigh(H)
    gt = Q.T @ g
    if w[0] > 0:
        d = Q @ (-gt / w)
        if np.linalg.norm(d) <= gam:
            return d, 0.0
    lo = max(0.0, -w[0]) + 1e-12
    nd = lambda lam: np.linalg.norm(gt / (w + lam))
    hi = lo + 1.0
    while nd(hi) > gam: hi *= 2
    for _ in range(200):                 # nd(lam) es decreciente en lam
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if nd(mid) > gam else (lo, mid)
    lam = 0.5*(lo + hi)
    return Q @ (-gt / (w + lam)), lam

def region_confianza(x0, gam0, iters=2, s1=0.1, s2=0.75, b1=0.5, b2=2.0):
    x, gam = np.array(x0, float), gam0
    for k in range(iters):
        g, H = grad(x), hess(x)
        print(f"\n" + "="*20 + " Iteracion {k} " + "="*20)
        print(f"x^{k} = {x},\tf = {f(x):.6f},\ngrad\t = {g},\nH\t = diag{np.diag(H)},\ngamma\t = {gam}")
        print("\t"+"-"*30)
        while True:
            d, lam = subproblema(g, H, gam)
            pred = -(g @ d + 0.5 * d @ H @ d)          # f_k(0)-f_k(d)
            real = f(x) - f(x + d)
            r = real / pred
            print(f"gamma\t = {gam:.6f}\t\t\tlambda\t = {lam:.6f}\nd\t = {d}\t||d||\t = {np.linalg.norm(d):.6f}")
            print(f"\nf(x+d)\t = {f(x+d):.6f}\nreal\t = {real:.6f}\tpred = {pred:.6f}\tr = {r:.6f}")
            if r >= s1: break
            gam = b1 * np.linalg.norm(d); print(" rechazado, nuevo gamma =", gam)
        xn = x + d
        if r >= s2 and np.isclose(np.linalg.norm(d), gam): gam *= b2
        x = xn
        print(f"\nx^{k+1}\t = {x},\tf = {f(x):.6f},\tgamma_{k+1} = {gam}")
    return x

region_confianza([2, 1], 1.0)
