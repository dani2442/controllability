import numpy as np
from scipy.integrate import solve_ivp

np.random.seed(42)
n = 2
m = 1
A = np.random.randn(n, n) - 2 * np.eye(n) # Make it stable
B = np.random.randn(n, m)
N = np.random.randn(n, n*m)

def u(t):
    return np.sin(t) + np.sin(np.pi * t) + np.sin(np.sqrt(2) * t) + np.sin(np.sqrt(3) * t) + np.sin(np.sqrt(5) * t)

def system(t, x):
    ut = u(t)
    return A @ x + B @ [ut] + N @ (np.kron(x, [ut]))

sol = solve_ivp(system, [0, 2], [1.0, 0.5], max_step=0.01)
t = sol.t
x = sol.y
ut = np.array([u(ti) for ti in t])
vt = np.vstack([ut, x[0]*ut, x[1]*ut])

print("Rank of V:", np.linalg.matrix_rank(vt @ vt.T), "expected", vt.shape[0])
