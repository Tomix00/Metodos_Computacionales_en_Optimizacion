import numpy as np
from fractions import Fraction

# f(x) = x1^2 + 2 x2^2 = 1/2 x^T A x - b^T x,  A = diag(2,4), b = 0
A = np.diag([2.0, 4.0]); b = np.zeros(2)
f    = lambda x: x[0]**2 + 2*x[1]**2
grad = lambda x: A @ x - b

def gradiente_conjugado(x0, tol=1e-12, maxit=None):
    x = np.array(x0, float); g = grad(x); d = -g
    maxit = maxit or len(x)                 # terminacion en <= n pasos
    print(f"k=0: x={x}, f={f(x):.6f}, g={g}, d={d}")
    for k in range(maxit):
        if np.linalg.norm(g) <= tol: break
        Ad    = A @ d
        alpha = (g @ g) / (d @ Ad)          # paso exacto
        x     = x + alpha * d
        g_new = g + alpha * Ad              # g_{k+1} = A x_{k+1} - b
        beta  = (g_new @ g_new) / (g @ g)   # Fletcher-Reeves
        d     = -g_new + beta * d
        print(f"\nk={k}: alpha={alpha:.6f} ({Fraction(alpha).limit_denominator(1000)}), beta_next={beta:.6f} ({Fraction(beta).limit_denominator(1000)})")
        print(f"k={k+1}: x={x}, f={f(x):.3e}, g={g_new}, d={d}")
        g = g_new
    return x

x = gradiente_conjugado([1, 1])
print("\nminimizador:", x)
