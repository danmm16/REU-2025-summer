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

def create_grouped_frequencies(N, n_groups, freq_spread=0.5, key=random.PRNGKey(42)):
    """
    Create natural frequencies with some oscillators sharing the same frequency
    
    Parameters:
    -----------
    N : int
        Total number of oscillators
    n_groups : int
        Number of oscillators that share the same frequency
    freq_spread : float
        Standard deviation of frequency distribution
    key : jax random key
        Random number generator key
    
    Returns:
    --------
    omega : array
        Natural frequencies with n_groups identical values
    group_indices : array
        Indices of oscillators that belong to the group
    """
    key1, key2, key3 = random.split(key, 3)
    
    # Create base frequency distribution
    omega = random.normal(key1, (N,)) * freq_spread
    
    if n_groups > 0 and n_groups <= N:
        # Choose which oscillators will share a frequency
        group_indices = random.choice(key2, N, (n_groups,), replace=False)
        
        # Generate a shared frequency for the group
        shared_freq = random.normal(key3, ()) * freq_spread
        
        # Assign the shared frequency to group members
        omega = omega.at[group_indices].set(shared_freq)
        
        return omega, group_indices
    else:
        return omega, jnp.array([])

def create_frequency_options(N, key):
    """
    Create different frequency distribution options
    """
    key1, key2, key3, key4, key5, key6, key7 = random.split(key, 7)
    
    options = {
        'all_different': (random.normal(key1, (N,)) * 0.5, jnp.array([])),
        'small_group': create_grouped_frequencies(N, n_groups=3, freq_spread=0.5, key=key2),
        'medium_group': create_grouped_frequencies(N, n_groups=N//3, freq_spread=0.5, key=key3),
        'large_group': create_grouped_frequencies(N, n_groups=N//2, freq_spread=0.5, key=key4),
        'larger_group': create_grouped_frequencies(N, n_groups=(2*N)//3, freq_spread=0.5, key=key5),
        'largest_group': create_grouped_frequencies(N, n_groups=N-3, freq_spread=0.5, key=key6),
        'all': create_grouped_frequencies(N, n_groups=N, freq_spread=0.5, key=key7),
    }
    
    return options

# Parameters
N = 20  # Number of oscillators
n_realizations = 30  # Multiple runs for averaging
n_steps_transient = 1500  # Steps to reach steady state
n_steps_measure = 750   # Steps for measurement

# Choose frequency distribution type
FREQUENCY_TYPE = 'medium_group'  # Options: 'all_different', 'small_group', 'medium_group', 'large_group', 'larger_group', 'largest_group', 'all'

print(f"Frequency distribution type: {FREQUENCY_TYPE}")

# Natural frequencies
key = random.PRNGKey(123)
omega_key, sim_key = random.split(key)

# Create frequency options
freq_options = create_frequency_options(N, omega_key)
omega, group_indices = freq_options[FREQUENCY_TYPE]

# Print frequency information
print(f"Number of oscillators: {N}")
if len(group_indices) > 0:
    print(f"Oscillators with shared frequency: {len(group_indices)} (indices: {group_indices})")
    print(f"Shared frequency value: {omega[group_indices[0]]:.3f}")
    print(f"Other frequencies range: [{jnp.min(omega):.3f}, {jnp.max(omega):.3f}]")
else:
    print("All oscillators have different frequencies")
    print(f"Frequency range: [{jnp.min(omega):.3f}, {jnp.max(omega):.3f}]")

print(f"Frequency standard deviation: {jnp.std(omega):.3f}")

# Coupling strength range
K_values = jnp.linspace(0, 8, 120)
order_params = []

print("\nComputing bifurcation diagram...")
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
plt.figure(figsize=(16, 10))

# Main bifurcation diagram
plt.subplot(2, 3, 1)
plt.plot(K_values, r_mean, 'b-', linewidth=2, label='Mean order parameter')
plt.fill_between(K_values, r_mean - r_std, r_mean + r_std, alpha=0.3, color='blue', label='±1 std')
plt.axvline(K_critical_theory, color='red', linestyle='--', alpha=0.7, 
           label=f'Theoretical K_c ≈ {K_critical_theory:.2f}')
plt.xlabel('Coupling strength K')
plt.ylabel('Order parameter r')
plt.title(f'Kuramoto Bifurcation: {FREQUENCY_TYPE.replace("_", " ").title()}')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 1)

# Zoom in on critical region
plt.subplot(2, 3, 2)
critical_region = (K_values > K_critical_theory - 1.0) & (K_values < K_critical_theory + 1.0)
if jnp.any(critical_region):
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
K_examples = [0.5, K_critical_theory, 6.0]
titles = ['Incoherent (K < K_c)', 'Near Critical Point', 'Synchronized (K > K_c)']

for idx, (K_ex, title) in enumerate(zip(K_examples, titles)):
    plt.subplot(2, 3, 4 + idx)
    
    # Simulate one example
    example_key = random.PRNGKey(999)
    phases_ex = simulate_kuramoto(omega, K_ex, n_steps_transient, key=example_key)
    
    # Plot on unit circle
    theta_circle = jnp.linspace(0, 2*jnp.pi, 100)
    plt.plot(jnp.cos(theta_circle), jnp.sin(theta_circle), 'lightgray', alpha=0.5)
    
    # Plot oscillators with different colors for grouped vs individual
    if len(group_indices) > 0:
        # Plot group members in red
        group_phases = phases_ex[group_indices]
        x_group = jnp.cos(group_phases)
        y_group = jnp.sin(group_phases)
        plt.scatter(x_group, y_group, c='red', s=50, alpha=0.8, 
                   label=f'Group ({len(group_indices)})', edgecolors='black', linewidth=0.5)
        
        # Plot individual oscillators in blue
        individual_mask = jnp.ones(N, dtype=bool).at[group_indices].set(False)
        individual_phases = phases_ex[individual_mask]
        x_individual = jnp.cos(individual_phases)
        y_individual = jnp.sin(individual_phases)
        plt.scatter(x_individual, y_individual, c='blue', s=30, alpha=0.6, 
                   label=f'Individual ({N-len(group_indices)})', edgecolors='black', linewidth=0.5)
    else:
        # All different frequencies - use colormap
        colors = plt.cm.hsv(jnp.linspace(0, 1, len(phases_ex)))
        x_pos = jnp.cos(phases_ex)
        y_pos = jnp.sin(phases_ex)
        plt.scatter(x_pos, y_pos, c=colors, s=30, alpha=0.8)
    
    # Plot order parameter vector
    r_ex = calculate_order_parameter(phases_ex)
    mean_phase = jnp.angle(jnp.mean(jnp.exp(1j * phases_ex)))
    plt.arrow(0, 0, r_ex * jnp.cos(mean_phase), r_ex * jnp.sin(mean_phase),
              head_width=0.05, head_length=0.05, fc='black', ec='black', linewidth=2)
    
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.gca().set_aspect('equal')
    plt.title(f'{title}\nK = {K_ex:.2f}, r = {r_ex:.3f}')
    plt.grid(True, alpha=0.3)
    if len(group_indices) > 0 and idx == 0:  # Add legend to first subplot only
        plt.legend(fontsize=8)

# Add frequency distribution subplot
plt.subplot(2, 3, 3)
plt.hist(omega, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
if len(group_indices) > 0:
    # Highlight the shared frequency
    shared_freq = omega[group_indices[0]]
    plt.axvline(shared_freq, color='red', linewidth=3, alpha=0.8, 
               label=f'Shared freq ({len(group_indices)} osc.)')
    plt.legend()
plt.xlabel('Natural frequency ω')
plt.ylabel('Number of oscillators')
plt.title('Frequency Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"\nSummary:")
print(f"Number of oscillators: {N}")
print(f"Frequency distribution: {FREQUENCY_TYPE}")
print(f"Natural frequency std: {jnp.std(omega):.3f}")
if len(group_indices) > 0:
    print(f"Group size: {len(group_indices)} oscillators")
    print(f"Shared frequency: {omega[group_indices[0]]:.3f}")
print(f"Final order parameter at K=0: {r_mean[0]:.3f}")
print(f"Final order parameter at max K: {r_mean[-1]:.3f}")

# Find empirical critical point (steepest increase)
dr_dk = jnp.diff(r_mean) / jnp.diff(K_values)
critical_idx = jnp.argmax(dr_dk)
K_critical_empirical = K_values[critical_idx]
print(f"Empirical critical coupling: {K_critical_empirical:.3f}")
print(f"Theory vs empirical error: {abs(K_critical_theory - K_critical_empirical):.3f}")

print(f"\nTo try different frequency distributions, change FREQUENCY_TYPE to:")
print("  'all_different' - All unique frequencies")
print("  'small_group'   - 3 oscillators share a frequency") 
print("  'medium_group'  - 1/3 of oscillators share a frequency")
print("  'large_group'   - 1/2 of oscillators share a frequency")
print("  'larger_group'  - 2/3 of oscillators share a frequency")
print("  'largest_group' - All but 3 oscillators share a frequency")
print("  'all'           - All oscillators have identical frequency")