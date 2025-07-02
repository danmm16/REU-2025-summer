import jax; jax.config.update("jax_enable_x64", True)
from jax import random, jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

@jit
def kuramoto_step(phases, omega, K, dt=0.1):
    """Single step of Kuramoto model integration"""
    N = len(phases)
    # Calculate coupling term: K/N * sum(sin(theta_j - theta_i))
    phase_diff = phases[:, None] - phases[None, :]  # Broadcasting for all pairs
    coupling = K/N * jnp.sum(jnp.sin(phase_diff), axis=1)
    
    # Update phases: dtheta/dt = omega + coupling
    new_phases = phases + dt * (omega + coupling)
    return new_phases % (2 * jnp.pi)  # Keep phases in [0, 2π]

def simulate_kuramoto(omega, K, n_steps=2000, dt=0.1, key=random.PRNGKey(42)):
    """Simulate Kuramoto model to steady state"""
    N = len(omega)
    # Random initial phases
    phases = random.uniform(key, (N,), maxval=2*jnp.pi)
    
    # Integrate to steady state
    for _ in range(n_steps):
        phases = kuramoto_step(phases, omega, K, dt)
    
    return phases

def calculate_order_parameter(phases):
    """Calculate order parameter r"""
    z = jnp.mean(jnp.exp(1j * phases))  # Complex order parameter
    return jnp.abs(z)  # Magnitude gives synchronization strength

# Parameters
N = 50  # Number of oscillators
n_realizations = 20  # Multiple runs for averaging
n_steps_transient = 1500  # Steps to reach steady state
n_steps_measure = 500   # Steps for measurement

# Natural frequencies from Lorentzian distribution (common in Kuramoto studies)
key = random.PRNGKey(123)
omega_key, sim_key = random.split(key)

# Generate natural frequencies with some spread
omega = random.normal(omega_key, (N,)) * 0.5  # Standard deviation of 0.5

# Coupling strength range
K_values = jnp.linspace(0, 4, 80)
order_params = []

print("Computing bifurcation diagram...")
print("This may take a moment...")

for i, K in enumerate(K_values):
    if i % 10 == 0:
        print(f"Progress: {i/len(K_values)*100:.1f}%")
    
    r_values = []
    
    # Multiple realizations for each K value
    for realization in range(n_realizations):
        # Different random seed for each realization
        real_key = random.PRNGKey(realization + 1000)
        
        # Simulate to steady state
        phases = simulate_kuramoto(omega, K, n_steps_transient, key=real_key)
        
        # Measure order parameter over several steps
        r_measures = []
        for _ in range(n_steps_measure):
            phases = kuramoto_step(phases, omega, K)
            r = calculate_order_parameter(phases)
            r_measures.append(r)
        
        # Average over measurement period
        r_avg = jnp.mean(jnp.array(r_measures))
        r_values.append(r_avg)
    
    # Store average and std over realizations
    order_params.append(r_values)

# Convert to arrays for plotting
order_params = jnp.array(order_params)
r_mean = jnp.mean(order_params, axis=1)
r_std = jnp.std(order_params, axis=1)

# Theoretical critical coupling (approximation for Lorentzian distribution)
# K_c ≈ 2 * γ where γ is half-width of frequency distribution
gamma = jnp.std(omega)
K_critical_theory = 2 * gamma

print(f"\nSimulation complete!")
print(f"Theoretical critical coupling K_c ≈ {K_critical_theory:.3f}")

# Create bifurcation plot
plt.figure(figsize=(14, 10))

# Main bifurcation diagram
plt.subplot(2, 2, 1)
plt.plot(K_values, r_mean, 'b-', linewidth=2, label='Mean order parameter')
plt.fill_between(K_values, r_mean - r_std, r_mean + r_std, alpha=0.3, color='blue', label='±1 std')
plt.axvline(K_critical_theory, color='red', linestyle='--', alpha=0.7, 
           label=f'Theoretical K_c ≈ {K_critical_theory:.2f}')
plt.xlabel('Coupling strength K')
plt.ylabel('Order parameter r')
plt.title('Kuramoto Model Bifurcation Diagram')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 1)

# Zoom in on critical region
plt.subplot(2, 2, 2)
critical_region = (K_values > K_critical_theory - 0.5) & (K_values < K_critical_theory + 0.5)
plt.plot(K_values[critical_region], r_mean[critical_region], 'b-', linewidth=2)
plt.fill_between(K_values[critical_region], 
                (r_mean - r_std)[critical_region], 
                (r_mean + r_std)[critical_region], alpha=0.3, color='blue')
plt.axvline(K_critical_theory, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Coupling strength K')
plt.ylabel('Order parameter r')
plt.title('Critical Region (Zoomed)')
plt.grid(True, alpha=0.3)

# Show phase snapshots at different coupling strengths
K_examples = [0.5, K_critical_theory, 3.0]
titles = ['Incoherent (K < K_c)', 'Near Critical Point', 'Synchronized (K > K_c)']

for idx, (K_ex, title) in enumerate(zip(K_examples, titles)):
    plt.subplot(2, 3, 4 + idx)
    
    # Simulate one example
    example_key = random.PRNGKey(999)
    phases_ex = simulate_kuramoto(omega, K_ex, n_steps_transient, key=example_key)
    
    # Plot on unit circle
    theta_circle = jnp.linspace(0, 2*jnp.pi, 100)
    plt.plot(jnp.cos(theta_circle), jnp.sin(theta_circle), 'lightgray', alpha=0.5)
    
    # Plot oscillators
    colors = plt.cm.hsv(jnp.linspace(0, 1, N))
    x_pos = jnp.cos(phases_ex)
    y_pos = jnp.sin(phases_ex)
    plt.scatter(x_pos, y_pos, c=colors, s=30, alpha=0.8)
    
    # Plot order parameter vector
    r_ex = calculate_order_parameter(phases_ex)
    mean_phase = jnp.angle(jnp.mean(jnp.exp(1j * phases_ex)))
    plt.arrow(0, 0, r_ex * jnp.cos(mean_phase), r_ex * jnp.sin(mean_phase),
              head_width=0.05, head_length=0.05, fc='red', ec='red', linewidth=2)
    
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.gca().set_aspect('equal')
    plt.title(f'{title}\nK = {K_ex:.2f}, r = {r_ex:.3f}')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"\nSummary:")
print(f"Number of oscillators: {N}")
print(f"Natural frequency std: {jnp.std(omega):.3f}")
print(f"Final order parameter at K=0: {r_mean[0]:.3f}")
print(f"Final order parameter at K=4: {r_mean[-1]:.3f}")

# Find empirical critical point (steepest increase)
dr_dk = jnp.diff(r_mean) / jnp.diff(K_values)
critical_idx = jnp.argmax(dr_dk)
K_critical_empirical = K_values[critical_idx]
print(f"Empirical critical coupling: {K_critical_empirical:.3f}")
print(f"Theory vs empirical error: {abs(K_critical_theory - K_critical_empirical):.3f}")