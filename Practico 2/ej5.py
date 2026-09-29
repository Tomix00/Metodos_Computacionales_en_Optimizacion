import numpy as np

Q = np.array([[5, 2, 1],
              [2, 7, 3],
              [1, 3, 9]], dtype=float)
c = np.array([-9, 0, -8], dtype=float)

x = np.array([0, 0, 0], dtype=float)
B = np.eye(3)

def gradf(x):
    return Q @ x - c

tol = 1e-6 # 10^-6
max_iter = 1000

print("=== Método BFGS ===")
print(f"x_0 = {x}\n")
print(f"H0 = \n{B}\n")

for k in range(max_iter):
    g = gradf(x)

    if np.linalg.norm(g) < tol:
        print(f"Convergencia alcanzada en la iteración {k}.")
        break

    # direccion
    d = np.linalg.solve(B, -g)

    # Busqueda lineal exacta para función cuadratica
    alpha = -np.dot(g,d) / np.dot(d, Q @ d)

    # Actualizacion del punto
    x_new = x + alpha * d
    s = x_new - x

    g_new = gradf(x_new)
    y = g_new - g

    print(f"\t\tIteración {k}:")
    print(f"Paso alpha: {alpha:.5f}")
    print(f"Dirección d: {d.round(4)}")
    print(f"Nuevo x: {x_new.round(4)}")
    print(f"Nuevo s: {s.round(4)}")
    print(f"Nuevo y: {y.round(4)}")
    print(f"Norma de g: {np.linalg.norm(g).round(4)}")

    # Atualizamos x
    x = x_new

    if np.linalg.norm(y) < tol:
        break

    # Actualizacion BFGS
    Bs = B @ s
    term2_num = np.outer(Bs, Bs)
    term2_den = s @ Bs
    term2 = term2_num / term2_den

    term3_num = np.outer(y,y)
    term3_den = np.dot(y, s)

    if term3_den <= 1e-10:
        print("Advertencia: y_k^T s_k es muy cercano a cero. La actualización podría ser inestable.")
        break

    term3 = term3_num / term3_den

    B_next = B - term2 + term3
    B = B_next
    print(f"Nuevo H: \n{B}\n")
    print("-"*50)
print(f"Solución final: x* = {x.round(6)}")