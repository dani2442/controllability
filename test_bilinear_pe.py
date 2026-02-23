import numpy as np
from scipy.integrate import solve_ivp

np.random.seed(42)
n = 2
m = 1
A = np.random.randn(n, n)
B = np.random.randn(n, m)
N = np.random.randn(n, n*m)

# Rich input
def u(t):
    return np.sin(t) + np.sin(np.pi * t) + np.sin(np.sqrt(2) * t) + np.sin(np.sqrt(3) * t) + np.sin(np.sqrt(5) * t)

def u_dot(t):
    return np.cos(t) + np.pi * np.cos(np.pi * t) + np.sqrt(2) * np.cos(np.sqrt(2) * t) + np.sqrt(3) * np.cos(np.sqrt(3) * t) + np.sqrt(5) * np.cos(np.sqrt(5) * t)

def system(t, x):
    ut = u(t)
    # x is n, u is m, N is n x (nm)
    return A @ x + B @ [ut] + N @ (np.kron(x, [ut]))

sol = solve_ivp(system, [0, 10], [1.0, 0.5], max_step=0.01)
t = sol.t
x = sol.y

ut = np.array([u(ti) for ti in t])
udot = np.array([u_dot(ti) for ti in t])
vt = np.vstack([ut, x[0]*ut, x[1]*ut])
vdot = np.gradient(vt, t, axis=1)

V_order1 = vt
V_order2 = np.vstack([vt, vdot])

print("Rank of V_order1:", np.linalg.matrix_rank(V_order1 @ V_order1.T), "expected", V_order1.shape[0])
print("Rank of V_order2:", np.linalg.matrix_rank(V_order2 @ V_order2.T), "expected", V_order2.shape[0])

# Check affine independence of x
X_aug = np.vstack([np.ones_like(t), x])
print("Rank of X_aug:", np.linalg.matrix_rank(X_aug @ X_aug.T), "expected", X_aug.shape[0])

