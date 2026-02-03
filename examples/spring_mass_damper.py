"""Spring-mass-damper example with animated visualization.

This example:
1. Simulates a spring-mass-damper system with sinusoidal input
2. Computes the Hautus estimation error for [0, t_n] as n increases
3. Creates an animated visualization with 3 panels:
   - Left: 2D phase portrait (position vs velocity)
   - Middle: Control input (top) and states (bottom)
   - Right: Error convergence as t_n increases
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples import save_fig

from control import (
    ControlledLinearSDE,
    create_time_grid,
    sdeint_safe,
    compare_with_true,
    gramian_Sz_time,
)


def create_spring_mass_damper(m=1.0, c=0.5, k=2.0, device="cpu"):
    """Create state-space matrices for spring-mass-damper system.
    
    The system is:
        m*x'' + c*x' + k*x = u
        
    In state-space form with state [position, velocity]:
        dx/dt = A*x + B*u
        
    where:
        A = [[0, 1], [-k/m, -c/m]]
        B = [[0], [1/m]]
    
    Args:
        m: Mass
        c: Damping coefficient  
        k: Spring constant
        device: Torch device
        
    Returns:
        A, B: System matrices
    """
    A = torch.tensor([
        [0.0, 1.0],
        [-k/m, -c/m]
    ], device=device)
    
    B = torch.tensor([
        [0.0],
        [1.0/m]
    ], device=device)
    
    return A, B


def sinusoidal_control(t, x, frequencies=[0.5, 1.2, 2.0], amplitudes=[1.0, 0.5, 0.3]):
    """Generate sinusoidal control input.
    
    u(t) = Σ a_i * sin(ω_i * t)
    
    Args:
        t: Time (scalar or tensor)
        x: State (unused, for interface compatibility)
        frequencies: List of frequencies
        amplitudes: List of amplitudes
        
    Returns:
        Control input tensor
    """
    device = x.device
    dtype = x.dtype
    batch_size = x.shape[0]
    
    # Convert to numpy for computation if tensor
    if isinstance(t, torch.Tensor):
        t_val = t.cpu().numpy() if t.ndim == 0 else t.cpu().numpy()
    else:
        t_val = t
    
    # Sum of sinusoids
    u_val = sum(a * np.sin(w * t_val) for w, a in zip(frequencies, amplitudes))
    
    if isinstance(t, torch.Tensor) and t.ndim > 0:
        # Time vector (N,) -> (N, batch, 1)
        u = torch.tensor(u_val, device=device, dtype=dtype)
        return u[:, None, None].expand(-1, batch_size, 1)
    else:
        # Scalar time -> (batch, 1)
        u = torch.tensor([[u_val]], device=device, dtype=dtype)
        return u.expand(batch_size, -1)


def compute_errors_over_time(x, u, A, B, dt, lam, t_indices):
    """Compute estimation errors for [0, t_n] for each n in t_indices.
    
    Args:
        x: Full state trajectory (N, n)
        u: Full input trajectory (N, m)
        A, B: True system matrices
        dt: Time step
        lam: Eigenvalue for Hautus test
        t_indices: List of time indices to evaluate
        
    Returns:
        errors: Array of errors for each t_n
    """
    errors = []
    
    for idx in t_indices:
        if idx < 10:  # Skip very short horizons
            errors.append(np.nan)
            continue
            
        x_n = x[:idx+1]
        u_n = u[:idx+1]
        
        try:
            result = compare_with_true(A, B, x_n, u_n, dt, lam, ridge=1e-8, method="time")
            errors.append(result["error_norm"])
        except Exception:
            errors.append(np.nan)
    
    return np.array(errors)


def main():
    # Set random seed
    torch.manual_seed(42)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Spring-mass-damper parameters
    m, c, k = 1.0, 0.3, 2.0
    A, B = create_spring_mass_damper(m, c, k, device)
    
    # Noise matrix (small noise for demonstration)
    n, m_input = A.shape[0], B.shape[1]
    q = 1  # Single noise source
    Beta = torch.tensor([[0.0], [0.1]], device=device)  # Noise on velocity
    
    # Simulation parameters
    T = 30.0  # 30 seconds of data
    dt = 0.01
    
    # Create SDE with custom sinusoidal control
    control_fn = lambda t, x: sinusoidal_control(t, x, 
                                                  frequencies=[0.5, 1.5, 3.0],
                                                  amplitudes=[2.0, 1.0, 0.5])
    
    sde = ControlledLinearSDE(A, B, Beta, control_fn=control_fn).to(device)
    
    # Create time grid and simulate
    ts = create_time_grid(T, dt, device)
    x0 = torch.tensor([[1.0, 0.0]], device=device)  # Start at position=1, velocity=0
    
    print(f"Simulating spring-mass-damper for T={T}s...")
    x = sdeint_safe(sde, x0, ts, dt)[:, 0, :]  # (N, 2)
    u = sde.u(ts, x)
    if u.ndim == 3:
        u = u[:, 0, :]  # (N, 1)
    
    # Convert to numpy for plotting
    ts_np = ts.cpu().numpy()
    x_np = x.cpu().numpy()
    u_np = u.cpu().numpy()
    
    print(f"Trajectory shape: {x_np.shape}")
    print(f"Control shape: {u_np.shape}")
    
    # Compute errors at different time horizons
    # Eigenvalue for Hautus test (use a stable eigenvalue)
    lam = 0.5 + 0.5j
    
    # Sample time indices for computing errors (more points for smooth curves)
    N = len(ts)
    n_error_points = 100
    error_indices = np.unique(np.logspace(np.log10(50), np.log10(N-1), n_error_points).astype(int))
    
    print(f"Computing errors for {len(error_indices)} time horizons...")
    errors = compute_errors_over_time(x, u, A, B, dt, lam, error_indices)
    t_error_values = ts_np[error_indices]
    
    # Remove NaN values for error plotting
    valid_mask = ~np.isnan(errors)
    t_valid = t_error_values[valid_mask]
    errors_valid = errors[valid_mask]
    
    # Animation frame indices (subset for smoother playback)
    n_frames = 150
    t_indices = np.unique(np.linspace(10, N-1, n_frames).astype(int))
    
    print(f"Valid error points: {len(errors_valid)}")
    
    # =========================================================================
    # Create animated visualization
    # =========================================================================
    
    # Set up figure with 3 columns
    fig = plt.figure(figsize=(16, 6))
    
    # Left: Phase portrait (position vs velocity)
    ax_phase = fig.add_subplot(1, 3, 1)
    
    # Middle: Two subplots (control on top, states on bottom)
    ax_control = fig.add_subplot(2, 3, 2)
    ax_states = fig.add_subplot(2, 3, 5)
    
    # Right: Error convergence
    ax_error = fig.add_subplot(1, 3, 3)
    
    # Initialize plots
    
    # Phase portrait
    phase_line, = ax_phase.plot([], [], 'b-', alpha=0.7, linewidth=1.5, label='Trajectory')
    phase_start, = ax_phase.plot([], [], 'go', markersize=10, label='Start', zorder=5)
    phase_current, = ax_phase.plot([], [], 'r*', markersize=15, label='Current', zorder=5)
    ax_phase.set_xlim(x_np[:, 0].min() - 0.5, x_np[:, 0].max() + 0.5)
    ax_phase.set_ylim(x_np[:, 1].min() - 0.5, x_np[:, 1].max() + 0.5)
    ax_phase.set_xlabel('Position $x$')
    ax_phase.set_ylabel('Velocity $\\dot{x}$')
    ax_phase.set_title('Phase Portrait')
    ax_phase.legend(loc='upper right')
    ax_phase.grid(True, alpha=0.3)
    
    # Control input
    control_line, = ax_control.plot([], [], 'C1-', linewidth=1.2)
    ax_control.set_xlim(0, T)
    ax_control.set_ylim(u_np.min() - 0.5, u_np.max() + 0.5)
    ax_control.set_ylabel('Control $u$')
    ax_control.set_title('Control Input')
    ax_control.grid(True, alpha=0.3)
    ax_control.set_xticklabels([])
    
    # States
    pos_line, = ax_states.plot([], [], 'b-', linewidth=1.2, label='Position $x$')
    vel_line, = ax_states.plot([], [], 'r-', linewidth=1.2, label='Velocity $\\dot{x}$')
    ax_states.set_xlim(0, T)
    ax_states.set_ylim(min(x_np.min(), -2), max(x_np.max(), 2))
    ax_states.set_xlabel('Time $t$ (s)')
    ax_states.set_ylabel('State')
    ax_states.set_title('State Evolution')
    ax_states.legend(loc='upper right')
    ax_states.grid(True, alpha=0.3)
    
    # Error convergence
    error_line, = ax_error.plot([], [], 'C2o-', markersize=3, linewidth=1.5)
    # Reference line for T^{-1/2}
    if len(t_valid) > 1:
        ref_t = np.linspace(t_valid[0], t_valid[-1], 100)
        ref_err = errors_valid[0] * np.sqrt(t_valid[0]) / np.sqrt(ref_t)
        ax_error.plot(ref_t, ref_err, 'k--', alpha=0.5, label='$\\mathcal{O}(T^{-1/2})$')
    ax_error.set_xscale('log')
    ax_error.set_yscale('log')
    ax_error.set_xlim(t_valid[0] * 0.8, T * 1.2)
    ax_error.set_ylim(errors_valid[~np.isnan(errors_valid)].min() * 0.5, 
                       errors_valid[~np.isnan(errors_valid)].max() * 2)
    ax_error.set_xlabel('Horizon $T$ (s)')
    ax_error.set_ylabel('Estimation Error')
    ax_error.set_title('Error Convergence')
    ax_error.legend(loc='upper right')
    ax_error.grid(True, alpha=0.3, which='both')
    
    # Time text
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    def init():
        phase_line.set_data([], [])
        phase_start.set_data([], [])
        phase_current.set_data([], [])
        control_line.set_data([], [])
        pos_line.set_data([], [])
        vel_line.set_data([], [])
        error_line.set_data([], [])
        time_text.set_text('')
        return phase_line, phase_start, phase_current, control_line, pos_line, vel_line, error_line, time_text
    
    def update(frame):
        # Get current time index
        idx = t_indices[frame]
        t_current = ts_np[idx]
        
        # Update phase portrait
        phase_line.set_data(x_np[:idx+1, 0], x_np[:idx+1, 1])
        phase_start.set_data([x_np[0, 0]], [x_np[0, 1]])
        phase_current.set_data([x_np[idx, 0]], [x_np[idx, 1]])
        
        # Update control
        control_line.set_data(ts_np[:idx+1], u_np[:idx+1, 0])
        
        # Update states
        pos_line.set_data(ts_np[:idx+1], x_np[:idx+1, 0])
        vel_line.set_data(ts_np[:idx+1], x_np[:idx+1, 1])
        
        # Update error (show all errors up to current frame)
        valid_up_to = np.where((t_error_values <= t_current) & valid_mask)[0]
        if len(valid_up_to) > 0:
            error_line.set_data(t_error_values[valid_up_to], errors[valid_up_to])
        
        # Update time text
        time_text.set_text(f'$T = {t_current:.2f}$ s')
        
        return phase_line, phase_start, phase_current, control_line, pos_line, vel_line, error_line, time_text
    
    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(fig, update, frames=len(t_indices),
                         init_func=init, blit=True, interval=50)
    
    # Save animation
    output_path = 'paper/images/spring_mass_damper_animation.gif'
    print(f"Saving animation to {output_path}...")
    anim.save(output_path, writer='pillow', fps=20)
    print("Animation saved!")
    
    # Also create a static final figure
    fig_static, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Left: Phase portrait
    axes[0].plot(x_np[:, 0], x_np[:, 1], 'b-', alpha=0.7, linewidth=1.5)
    axes[0].scatter(x_np[0, 0], x_np[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    axes[0].scatter(x_np[-1, 0], x_np[-1, 1], c='red', s=100, marker='*', label='End', zorder=5)
    axes[0].set_xlabel('Position $x$')
    axes[0].set_ylabel('Velocity $\\dot{x}$')
    axes[0].set_title('Phase Portrait')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Middle: Create sub-gridspec for control and states
    # We need to recreate for static version
    
    # Right: Error convergence
    axes[2].loglog(t_valid, errors_valid, 'C2o-', markersize=4, linewidth=1.5, label='Empirical')
    ref_t = np.linspace(t_valid[0], t_valid[-1], 100)
    ref_err = errors_valid[0] * np.sqrt(t_valid[0]) / np.sqrt(ref_t)
    axes[2].plot(ref_t, ref_err, 'k--', alpha=0.5, label='$\\mathcal{O}(T^{-1/2})$')
    axes[2].set_xlabel('Horizon $T$ (s)')
    axes[2].set_ylabel('Estimation Error')
    axes[2].set_title('Error Convergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, which='both')
    
    # Middle: Use the middle axes for combined plot
    ax_mid = axes[1]
    ax_mid.clear()
    
    # Create twin axis for control vs states
    ax_twin = ax_mid.twinx()
    
    l1 = ax_mid.plot(ts_np, x_np[:, 0], 'b-', linewidth=1.2, label='Position $x$')
    l2 = ax_mid.plot(ts_np, x_np[:, 1], 'r-', linewidth=1.2, label='Velocity $\\dot{x}$')
    l3 = ax_twin.plot(ts_np, u_np[:, 0], 'C1--', linewidth=1.0, alpha=0.7, label='Control $u$')
    
    ax_mid.set_xlabel('Time $t$ (s)')
    ax_mid.set_ylabel('State')
    ax_twin.set_ylabel('Control', color='C1')
    ax_twin.tick_params(axis='y', labelcolor='C1')
    ax_mid.set_title('States and Control')
    
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax_mid.legend(lines, labels, loc='upper right')
    ax_mid.grid(True, alpha=0.3)
    
    plt.tight_layout()
    static_path = 'paper/images/spring_mass_damper_static.pdf'
    plt.savefig(static_path, dpi=150, bbox_inches='tight')
    print(f"Static figure saved to {static_path}")
    
    # Create the proper 3-panel figure with middle having 2 subplots
    fig2 = plt.figure(figsize=(16, 6))
    
    # Use GridSpec for better control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig2, height_ratios=[1, 1])
    
    # Left: Phase portrait (spans both rows)
    ax_left = fig2.add_subplot(gs[:, 0])
    ax_left.plot(x_np[:, 0], x_np[:, 1], 'b-', alpha=0.7, linewidth=1.5)
    ax_left.scatter(x_np[0, 0], x_np[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax_left.scatter(x_np[-1, 0], x_np[-1, 1], c='red', s=100, marker='*', label='End', zorder=5)
    ax_left.set_xlabel('Position $x$')
    ax_left.set_ylabel('Velocity $\\dot{x}$')
    ax_left.set_title('Phase Portrait')
    ax_left.legend()
    ax_left.grid(True, alpha=0.3)
    
    # Middle top: Control
    ax_mid_top = fig2.add_subplot(gs[0, 1])
    ax_mid_top.plot(ts_np, u_np[:, 0], 'C1-', linewidth=1.2)
    ax_mid_top.set_ylabel('Control $u$')
    ax_mid_top.set_title('Control Input')
    ax_mid_top.grid(True, alpha=0.3)
    ax_mid_top.set_xticklabels([])
    
    # Middle bottom: States
    ax_mid_bot = fig2.add_subplot(gs[1, 1])
    ax_mid_bot.plot(ts_np, x_np[:, 0], 'b-', linewidth=1.2, label='Position $x$')
    ax_mid_bot.plot(ts_np, x_np[:, 1], 'r-', linewidth=1.2, label='Velocity $\\dot{x}$')
    ax_mid_bot.set_xlabel('Time $t$ (s)')
    ax_mid_bot.set_ylabel('State')
    ax_mid_bot.set_title('State Evolution')
    ax_mid_bot.legend(loc='upper right')
    ax_mid_bot.grid(True, alpha=0.3)
    
    # Right: Error convergence (spans both rows)
    ax_right = fig2.add_subplot(gs[:, 2])
    ax_right.loglog(t_valid, errors_valid, 'C2o-', markersize=4, linewidth=1.5, label='Empirical')
    ref_t = np.linspace(t_valid[0], t_valid[-1], 100)
    ref_err = errors_valid[0] * np.sqrt(t_valid[0]) / np.sqrt(ref_t)
    ax_right.plot(ref_t, ref_err, 'k--', alpha=0.5, label='$\\mathcal{O}(T^{-1/2})$')
    ax_right.set_xlabel('Horizon $T$ (s)')
    ax_right.set_ylabel('Estimation Error')
    ax_right.set_title('Error Convergence')
    ax_right.legend()
    ax_right.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    proper_path = 'paper/images/spring_mass_damper.pdf'
    plt.savefig(proper_path, dpi=150, bbox_inches='tight')
    print(f"Proper 3-panel figure saved to {proper_path}")
    
    # Only show if display is available
    if os.environ.get('DISPLAY'):
        plt.show()
    else:
        plt.close('all')
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"System: Spring-mass-damper (m={m}, c={c}, k={k})")
    print(f"State dimension: n={n}")
    print(f"Input dimension: m={m_input}")
    print(f"Simulation time: T={T}s")
    print(f"Time step: dt={dt}")
    print(f"Number of samples: {len(ts)}")
    
    # Estimate convergence rate
    if len(errors_valid) > 10:
        log_t = np.log(t_valid)
        log_err = np.log(errors_valid)
        slope, _ = np.polyfit(log_t, log_err, 1)
        print(f"Estimated convergence rate: T^{{{slope:.3f}}}")
        print(f"Expected rate: T^{{-0.5}}")


if __name__ == "__main__":
    main()
