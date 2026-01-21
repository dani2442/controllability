"""Example comparing time-domain and FFT-based Hautus test methods.

This example shows:
1. Comparison of both methods for various λ values
2. Effect of frequency cutoff on FFT method
3. Computational comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples import save_fig

from control import (
    ControlledLinearSDE,
    create_time_grid,
    sdeint_safe,
    cross_moment_H_time,
    cross_moment_H_fft,
    gramian_Sz_time,
    estimate_Hautus_matrix,
    true_Hautus_matrix,
    make_stable_A,
    plot_matrix_comparison,
)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(789)
    
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # System dimensions
    n, m, q = 8, 4, 2
    
    # Create system matrices
    A = make_stable_A(n, device, margin=0.15)
    B = torch.randn(n, m, device=device)
    Beta = torch.randn(n, q, device=device) / n
    
    # Simulation parameters
    T = 100.0
    dt = 0.02  # Fine time step for FFT accuracy
    batch = 1
    
    # Create SDE and simulate
    sde = ControlledLinearSDE(A, B, Beta).to(device)
    ts = create_time_grid(T, dt, device)
    x0 = torch.zeros(batch, n, device=device)
    
    x = sdeint_safe(sde, x0, ts, dt)[:, 0, :]
    u = sde.u(ts, x)
    if u.ndim == 3:
        u = u[:, 0, :]
    
    print(f"Simulated trajectory: x.shape = {x.shape}")
    print(f"Number of time steps: {len(ts)}")
    
    # Test multiple λ values
    lambda_values = [
        0.0 + 0.0j,
        0.5 + 0.0j,
        0.0 + 1.0j,
        0.3 + 0.8j,
        -0.5 + 0.5j,
        1.0 + 1.5j,
    ]
    
    print(f"\n{'='*70}")
    print("COMPARISON OF TIME-DOMAIN AND FFT METHODS")
    print("="*70)
    print(f"{'λ':^15} {'σ_min(True)':^12} {'σ_min(Time)':^12} {'σ_min(FFT)':^12} {'Err Time':^10} {'Err FFT':^10}")
    print("-"*70)
    
    errors_time = []
    errors_fft = []
    sigma_mins_true = []
    sigma_mins_time = []
    sigma_mins_fft = []
    
    for lam in lambda_values:
        # True Hautus matrix
        P_true = true_Hautus_matrix(A, B, lam)
        sigma_true = torch.linalg.svdvals(P_true).min().item()
        sigma_mins_true.append(sigma_true)
        
        # Time-domain method
        H_time = cross_moment_H_time(x, u, lam=lam, dt=dt)
        Sz_time = gramian_Sz_time(x, u, dt=dt)
        P_hat_time = estimate_Hautus_matrix(H_time, Sz_time, ridge=1e-10)
        sigma_time = torch.linalg.svdvals(P_hat_time).min().item()
        err_time = torch.linalg.norm(P_hat_time - P_true.to(P_hat_time.dtype), ord=2).item()
        sigma_mins_time.append(sigma_time)
        errors_time.append(err_time)
        
        # FFT method
        H_fft, Sz_fft = cross_moment_H_fft(x, u, lam=lam, dt=dt, omega_max=200)
        P_hat_fft = estimate_Hautus_matrix(H_fft, Sz_fft, ridge=1e-10)
        sigma_fft = torch.linalg.svdvals(P_hat_fft).min().item()
        err_fft = torch.linalg.norm(P_hat_fft - P_true.to(P_hat_fft.dtype), ord=2).item()
        sigma_mins_fft.append(sigma_fft)
        errors_fft.append(err_fft)
        
        print(f"{str(lam):^15} {sigma_true:^12.4f} {sigma_time:^12.4f} {sigma_fft:^12.4f} {err_time:^10.4f} {err_fft:^10.4f}")
    
    # Effect of frequency cutoff
    print(f"\n{'='*70}")
    print("EFFECT OF FREQUENCY CUTOFF (ω_max) ON FFT METHOD")
    print("="*70)
    
    lam_test = 0.3 + 0.8j
    P_true = true_Hautus_matrix(A, B, lam_test)
    
    omega_max_values = [10, 20, 50, 100, 200, None]
    
    print(f"Testing λ = {lam_test}")
    print(f"{'ω_max':^12} {'σ_min(FFT)':^15} {'Error':^15}")
    print("-"*45)
    
    omega_errors = []
    omega_sigmas = []
    
    for omega_max in omega_max_values:
        H_fft, Sz_fft = cross_moment_H_fft(x, u, lam=lam_test, dt=dt, omega_max=omega_max)
        P_hat_fft = estimate_Hautus_matrix(H_fft, Sz_fft, ridge=1e-10)
        sigma_fft = torch.linalg.svdvals(P_hat_fft).min().item()
        err_fft = torch.linalg.norm(P_hat_fft - P_true.to(P_hat_fft.dtype), ord=2).item()
        
        omega_str = str(omega_max) if omega_max is not None else "None (all)"
        print(f"{omega_str:^12} {sigma_fft:^15.6f} {err_fft:^15.6f}")
        
        omega_errors.append(err_fft)
        omega_sigmas.append(sigma_fft)
    
    # Computational time comparison
    print(f"\n{'='*70}")
    print("COMPUTATIONAL TIME COMPARISON")
    print("="*70)
    
    n_repeats = 20
    
    # Time-domain timing
    start = time.time()
    for _ in range(n_repeats):
        H_time = cross_moment_H_time(x, u, lam=lam_test, dt=dt)
        Sz_time = gramian_Sz_time(x, u, dt=dt)
        P_hat = estimate_Hautus_matrix(H_time, Sz_time, ridge=1e-10)
    time_domain_elapsed = (time.time() - start) / n_repeats
    
    # FFT timing
    start = time.time()
    for _ in range(n_repeats):
        H_fft, Sz_fft = cross_moment_H_fft(x, u, lam=lam_test, dt=dt, omega_max=200)
        P_hat = estimate_Hautus_matrix(H_fft, Sz_fft, ridge=1e-10)
    fft_elapsed = (time.time() - start) / n_repeats
    
    print(f"Time-domain method: {time_domain_elapsed*1000:.3f} ms per call")
    print(f"FFT method:         {fft_elapsed*1000:.3f} ms per call")
    print(f"Speedup:            {time_domain_elapsed/fft_elapsed:.2f}x")
    
    # Visualization
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Error comparison bar chart
    ax = axs[0, 0]
    x_pos = np.arange(len(lambda_values))
    width = 0.35
    ax.bar(x_pos - width/2, errors_time, width, label='Time-domain')
    ax.bar(x_pos + width/2, errors_fft, width, label='FFT')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(lam) for lam in lambda_values], rotation=45)
    ax.set_ylabel('$\\|\\hat{P}_\\lambda - P_\\lambda\\|_2$')
    ax.set_title('Estimation Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Singular value comparison
    ax = axs[0, 1]
    ax.bar(x_pos - width, sigma_mins_true, width, label='True')
    ax.bar(x_pos, sigma_mins_time, width, label='Time-domain')
    ax.bar(x_pos + width, sigma_mins_fft, width, label='FFT')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(lam) for lam in lambda_values], rotation=45)
    ax.set_ylabel('$\\sigma_{\\min}$')
    ax.set_title('Minimum Singular Value Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Frequency cutoff effect
    ax = axs[1, 0]
    omega_labels = [str(w) if w is not None else 'None' for w in omega_max_values]
    ax.bar(range(len(omega_max_values)), omega_errors)
    ax.set_xticks(range(len(omega_max_values)))
    ax.set_xticklabels(omega_labels)
    ax.set_xlabel('$\\omega_{\\max}$')
    ax.set_ylabel('Error $\\|\\hat{P} - P\\|_2$')
    ax.set_title('Effect of Frequency Cutoff on FFT Method')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Matrix visualization for one λ
    ax = axs[1, 1]
    lam = lambda_values[3]  # 0.3 + 0.8j
    P_true = true_Hautus_matrix(A, B, lam)
    H_time = cross_moment_H_time(x, u, lam=lam, dt=dt)
    Sz_time = gramian_Sz_time(x, u, dt=dt)
    P_hat_time = estimate_Hautus_matrix(H_time, Sz_time, ridge=1e-10)
    
    diff = (P_hat_time - P_true.to(P_hat_time.dtype)).abs()
    im = ax.imshow(diff.cpu().numpy(), cmap='hot', aspect='auto')
    ax.set_title(f'$|\\hat{{P}}_\\lambda - P_\\lambda|$ for $\\lambda = {lam}$')
    ax.set_xlabel('Column index')
    ax.set_ylabel('Row index')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    p = save_fig('method_comparison.png', dpi=150)
    print(f"\nPlot saved to '{p}'")
    plt.show()


if __name__ == "__main__":
    main()
