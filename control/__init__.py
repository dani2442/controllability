import torchsde
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Union, Tuple

class ControlledLinearSDE(torchsde.SDEIto):
    def __init__(self, A: torch.Tensor, B: torch.Tensor, Beta: torch.tensor):
        super().__init__(noise_type="general")  # dW has same dim as x
        self.A = A                                # (n, n)
        self.B = B                                # (n, m)
        self.Beta = Beta                          # (n, q)

    def u(self, t: torch.Tensor, batch_size, m) -> torch.Tensor:
        w = torch.linspace(1, 5, m)
        u = torch.sin(t[..., None] * w)                            # (m,) or (N,m)

        if t.ndim == 0:                                            # scalar -> (batch,m)
            return u.unsqueeze(0).expand(batch_size, -1)
        else:                                                      # (N,) -> (N,batch,m)
            return u.unsqueeze(1).expand(-1, batch_size, -1)

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Drift: f(t,x) = A x + B u(t)
        x shape: (batch, n)
        """
        # A x
        batch_size, _ = x.shape
        ut = self.u(t, batch_size, m).to(x)   # (batch, m)
        Ax = x @ self.A.T                            # (batch, n)
        But = ut @ self.B.T                          # (batch, n)
        return Ax + But

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Diffusion: g(t,x) = Beta  (additive noise)
        For noise_type="general", must return (batch, n, q).
        """
        batch_size, _ = x.shape
        return self.Beta.unsqueeze(0).expand(batch_size, -1, -1).to(x)
    

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

T = 100.0
dt = 0.1
n, m, q = 10, 5, 2
batch = 1

A = torch.randn(n, n, device=device)
A = A - (torch.linalg.eigvals(A).real.max() + 0.1) * torch.eye(n, device=device, dtype=A.dtype)

B = torch.randn(n, m, device=device)
Beta = torch.randn((n,q))/n

sde = ControlledLinearSDE(A, B, Beta).to(device)


ts = torch.linspace(0.0, T, int(T/dt)+1, device=device)  # time grid
x0 = torch.zeros(batch, n, device=device)            # initial condition

# Simulate sample path (Euler–Maruyama)
x = torchsde.sdeint(sde, x0, ts, method="euler", dt_min=dt)[:,0]  # (len(ts), batch, n)
u = sde.u(ts, batch, m)[:,0]
print("x.shape =", x.shape)        # (len(ts), batch, n)
print("u.shape =", u.shape)        # (len(ts), batch, n


blues = plt.cm.hsv(np.linspace(0.3, 1.0, n//2))
for idx, i in enumerate(range(0,n,2)):
  plt.plot(x[:, i].cpu().numpy(), x[:, i+1].cpu().numpy(),
             color=blues[idx], alpha=0.5)
  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate different shades of blue
blues = plt.cm.hsv(np.linspace(0.3, 1.0, n//3))

for i in range(n//3):
    ax.plot(x[:, 3*i].cpu().numpy(),
            x[:, 3*i+1].cpu().numpy(),
            x[:, 3*i+2].cpu().numpy(),
            color=blues[i], alpha=0.5)

plt.show()