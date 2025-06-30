import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq

def duffing_ode(t, y, c, gamma=1):
    """
    Duffing oscillator: x'' + x = cx^3
    State vector: y = [x, x']
    Returns: [x', x'' = -x + cx^3]
    """
    x, x_dot = y
    x_ddot = -2 * gamma * x_dot - x + c * x**3
    return [x_dot, x_ddot]

def solve_duffing(c, t_span=(0, 20), t_eval=None):
    """
    Solve the Duffing oscillator with initial conditions x(0)=1, x'(0)=0
    
    Parameters:
    c: nonlinearity parameter
    t_span: time span for integration
    t_eval: specific time points to evaluate (if None, uses default spacing)
    
    Returns:
    sol: solution object from solve_ivp
    """
    # Initial conditions: x(0) = 1, x'(0) = 0
    y0 = [1.0, 0.0]
    
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 2000)
    
    # Solve the ODE using RK45 method
    sol = solve_ivp(duffing_ode, t_span, y0, t_eval=t_eval, 
                    args=(c,), method='RK45', rtol=1e-8, atol=1e-10)
    
    return sol

def compute_power_spectrum(t, x):
    """
    Compute the power spectrum of the solution
    
    Parameters:
    t: time array
    x: displacement array
    
    Returns:
    freqs: frequency array
    power: power spectrum
    """
    # Ensure uniform time spacing for FFT
    dt = t[1] - t[0]
    n = len(x)
    
    # Compute FFT
    X = fft(x)
    freqs = fftfreq(n, dt)
    
    # Power spectrum (magnitude squared)
    power = np.abs(X)**2
    
    # Only return positive frequencies
    pos_mask = freqs >= 0
    return freqs[pos_mask], power[pos_mask]

def analyze_duffing_oscillator():
    """
    Complete analysis of the Duffing oscillator for different nonlinearity parameters
    """
    # Parameters to analyze
    c_values = [0.5, 3.0, -3.0]
    c_labels = ['Somewhat Nonlinear (c=0.5)', 'Strongly Nonlinear (c=3.0)', 'Softening Spring (Nonlinear c=-3.0)']
    colors = ['orange', 'blue', 'green']
    
    # Time span for analysis
    t_span = (0, 200)
    t_eval = np.linspace(0, 200, 20000)  # High resolution for good FFT
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Duffing Oscillator Analysis: x\'\' + x = cx³', fontsize=16, fontweight='bold')
    
    # Storage for solutions
    solutions = {}
    
    for i, (c, label, color) in enumerate(zip(c_values, c_labels, colors)):
        print(f"Solving for {label}...")
        
        # Solve the ODE
        sol = solve_duffing(c, t_span, t_eval)
        solutions[c] = sol
        
        # Extract solution
        t = sol.t
        x = sol.y[0]  # displacement
        x_dot = sol.y[1]  # velocity
        
        # Plot time series (top row)
        axes[0, i].plot(t, x, color=color, linewidth=2)
        axes[0, i].set_title(f'{label}\nTime Series', fontweight='bold')
        axes[0, i].set_xlabel('Time (t)')
        axes[0, i].set_ylabel('Displacement x(t)')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_xlim(0, 20)
        
        # Compute and plot power spectrum (bottom row)
        freqs, power = compute_power_spectrum(t, x)
        
        # Normalize power spectrum
        power_normalized = power / np.max(power)
        
        axes[1, i].semilogy(freqs, power_normalized, color=color, linewidth=2)
        axes[1, i].set_title(f'{label}\nPower Spectrum', fontweight='bold')
        axes[1, i].set_xlabel('Frequency (Hz)')
        axes[1, i].set_ylabel('Normalized Power')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_xlim(0, 5)  # Focus on low frequencies
        axes[1, i].set_ylim(1e-6, 1)
        
        # Add text with fundamental frequency
        fundamental_freq = freqs[np.argmax(power[1:])+1]  # Skip DC component
        axes[1, i].axvline(fundamental_freq, color='black', linestyle='--', alpha=0.7)
        axes[1, i].text(0.02, 0.95, f'f₀ ≈ {fundamental_freq:.3f} Hz', 
                       transform=axes[1, i].transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Phase portraits comparison
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for i, (c, label, color) in enumerate(zip(c_values, c_labels, colors)):
        sol = solutions[c]
        x = sol.y[0]
        x_dot = sol.y[1]
        
        ax.plot(x, x_dot, color=color, linewidth=2, label=label, alpha=0.8)
    
    ax.set_title('Phase Portraits Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Displacement x')
    ax.set_ylabel('Velocity x\'')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # Print harmonic analysis
    print("\n" + "="*60)
    print("HARMONIC ANALYSIS")
    print("="*60)
    
    for c, label in zip(c_values, c_labels):
        sol = solutions[c]
        t = sol.t
        x = sol.y[0]
        
        freqs, power = compute_power_spectrum(t, x)
        
        # Find peaks in power spectrum
        power_normalized = power / np.max(power)
        
        # Find significant harmonics (power > 1% of maximum)
        significant_indices = np.where(power_normalized > 0.01)[0]
        significant_freqs = freqs[significant_indices]
        significant_powers = power_normalized[significant_indices]
        
        # Sort by power
        sorted_indices = np.argsort(significant_powers)[::-1]
        
        print(f"\n{label}:")
        print(f"Significant frequency components (>1% of peak power):")
        for j, idx in enumerate(sorted_indices[:5]):  # Top 5 components
            freq = significant_freqs[idx]
            power_pct = significant_powers[idx] * 100
            if freq > 0.01:  # Skip near-DC components
                print(f"  f = {freq:.3f} Hz (Power: {power_pct:.1f}%)")

if __name__ == "__main__":
    # Run the complete analysis
    analyze_duffing_oscillator()
    
    # Additional detailed analysis for educational purposes
    print("\n" + "="*60)
    print("EDUCATIONAL INSIGHTS")
    print("="*60)
    print("""
Key Observations:

1. LINEAR CASE (c=0):
   - Pure sinusoidal oscillation at single frequency
   - Power spectrum shows single peak at fundamental frequency
   - Frequency ≈ 1/(2π) ≈ 0.159 Hz (natural frequency of harmonic oscillator)

2. WEAKLY NONLINEAR (c=0.1):
   - Nearly sinusoidal but with slight distortion
   - Small harmonic generation at odd multiples of fundamental
   - Phase portrait shows slight deviation from perfect ellipse

3. STRONGLY NONLINEAR (c=1.0):
   - Significantly distorted waveform
   - Strong harmonic generation (3rd, 5th harmonics visible)
   - Phase portrait shows pronounced nonlinear behavior
   - Frequency shift due to amplitude-dependent restoring force

Physical Interpretation:
- The x³ term represents a "hardening spring" (restoring force increases with amplitude)
- Larger c values lead to more harmonic distortion
- Energy spreads from fundamental frequency to higher harmonics
- This is a classic example of how nonlinearity generates frequency content
""")