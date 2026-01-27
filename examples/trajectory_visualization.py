"""Example demonstrating trajectory visualizations.

This example creates various visualizations of the SDE trajectories:
- 2D phase portraits
- 3D phase portraits
- Time series plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples import save_fig

from control import (
    ControlledLinearSDE,
    create_time_grid,
    sdeint_safe,
    make_stable_A,
    plot_trajectory_2d,
    plot_trajectory_3d,
    plot_time_series,
)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(2024)
    
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # System dimensions
    n, m, q = 9, 4, 2  # 9 states for nice 3D groupings
    
    # Create system matrices with different stability margins
    A = make_stable_A(n, device, margin=0.05)  # Slower decay
    B = torch.randn(n, m, device=device)
    Beta = torch.randn(n, q, device=device) / np.sqrt(n)
    
    # Simulation parameters
    T = 50.0
    dt = 0.02
    batch = 1
    
    # Create SDE and simulate
    sde = ControlledLinearSDE(A, B, Beta).to(device)
    ts = create_time_grid(T, dt, device)
    x0 = torch.randn(batch, n, device=device)  # Non-zero initial condition
    
    x = sdeint_safe(sde, x0, ts, dt)[:, 0, :]
    u = sde.u(ts, x)
    if u.ndim == 3:
        u = u[:, 0, :]
    
    print(f"Simulated trajectory: x.shape = {x.shape}")
    print(f"Time horizon: T = {T}")
    print(f"Time step: dt = {dt}")
    print(f"Number of steps: {len(ts)}")
    
    # Create visualizations
    
    # 1. 2D Phase Portraits
    fig1, ax1 = plot_trajectory_2d(
        x,
        pairs=[(0, 1), (2, 3), (4, 5), (6, 7)],
        title="2D Phase Portraits",
        cmap="tab10",
        alpha=0.6,
        figsize=(10, 8),
    )
    ax1.set_xlabel("$x_i$")
    ax1.set_ylabel("$x_j$")
    
    # 2. 3D Phase Portrait
    fig2, ax2 = plot_trajectory_3d(
        x,
        triples=[(0, 1, 2), (3, 4, 5), (6, 7, 8)],
        title="3D Phase Portraits",
        cmap="viridis",
        alpha=0.7,
        figsize=(12, 9),
    )
    
    # 3. Time Series - States
    fig3, ax3 = plot_time_series(
        ts,
        x,
        components=list(range(min(6, n))),
        title="State Evolution Over Time",
        ylabel="State $x_i(t)$",
        cmap="tab10",
        figsize=(14, 6),
    )
    
    # 4. Time Series - Inputs
    fig4, ax4 = plot_time_series(
        ts,
        u,
        components=list(range(m)),
        title="Control Input Over Time",
        ylabel="Input $u_j(t)$",
        cmap="Set2",
        figsize=(14, 5),
    )
    
    # 5. Multi-panel figure
    fig5, axs = plt.subplots(2, 3, figsize=(16, 10))
    
    # First row: 2D phase portraits
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 3))
    for idx, (i, j) in enumerate([(0, 1), (2, 3), (4, 5)]):
        axs[0, idx].plot(x[:, i].cpu(), x[:, j].cpu(), color=colors[idx], alpha=0.7)
        axs[0, idx].scatter(x[0, i].cpu(), x[0, j].cpu(), c='green', s=100, 
                           marker='o', label='Start', zorder=5)
        axs[0, idx].scatter(x[-1, i].cpu(), x[-1, j].cpu(), c='red', s=100, 
                           marker='*', label='End', zorder=5)
        axs[0, idx].set_xlabel(f"$x_{{{i}}}$")
        axs[0, idx].set_ylabel(f"$x_{{{j}}}$")
        axs[0, idx].set_title(f"Phase: $x_{{{i}}}$ vs $x_{{{j}}}$")
        axs[0, idx].legend(loc='best')
        axs[0, idx].grid(True, alpha=0.3)
    
    # Second row: time series
    for idx, i in enumerate([0, 3, 6]):
        axs[1, idx].plot(ts.cpu(), x[:, i].cpu(), 'b-', alpha=0.8, label=f'$x_{{{i}}}$')
        axs[1, idx].plot(ts.cpu(), x[:, i+1].cpu(), 'r-', alpha=0.8, label=f'$x_{{{i+1}}}$')
        axs[1, idx].plot(ts.cpu(), x[:, i+2].cpu(), 'g-', alpha=0.8, label=f'$x_{{{i+2}}}$')
        axs[1, idx].set_xlabel("Time $t$")
        axs[1, idx].set_ylabel("State")
        axs[1, idx].set_title(f"States $x_{{{i}}}, x_{{{i+1}}}, x_{{{i+2}}}$")
        axs[1, idx].legend(loc='best')
        axs[1, idx].grid(True, alpha=0.3)
    
    plt.suptitle(f"SDE Trajectory Visualization (n={n}, m={m}, T={T})", fontsize=14)
    plt.tight_layout()
    
    # 6. State energy over time
    fig6, ax6 = plt.subplots(figsize=(12, 5))
    
    energy = (x ** 2).sum(dim=1)
    ax6.plot(ts.cpu(), energy.cpu(), 'b-', linewidth=1.5)
    ax6.fill_between(ts.cpu().numpy(), 0, energy.cpu().numpy(), alpha=0.3)
    ax6.set_xlabel("Time $t$")
    ax6.set_ylabel("$\\|x(t)\\|_2^2$")
    ax6.set_title("State Energy Over Time")
    ax6.grid(True, alpha=0.3)
    
    # Save figures into images/ using shared helper
    # p1 = save_fig('trajectory_2d.png', fig=fig1, dpi=150)
    p2 = save_fig('paper/images/trajectory_3d.pdf', fig=fig2, dpi=150)
    # p3 = save_fig('trajectory_multipanel.png', fig=fig5, dpi=150)
    # p4 = save_fig('trajectory_energy.png', fig=fig6, dpi=150)

    print("\nFigures saved:")
    # print(f"  - {p1}")
    print(f"  - {p2}")
    # print(f"  - {p3}")
    # print(f"  - {p4}")
    
    plt.show()


if __name__ == "__main__":
    main()
