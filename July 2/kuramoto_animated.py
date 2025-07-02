import jax; jax.config.update("jax_enable_x64", True)
from jax import random, jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

@jit
def kuramoto_step(phases, omega, K, dt=0.05):
    """Single step of Kuramoto model integration"""
    N = len(phases)
    # Calculate coupling term: K/N * sum(sin(theta_j - theta_i))
    phase_diff = phases[:, None] - phases[None, :]  # Broadcasting for all pairs
    coupling = K/N * jnp.sum(jnp.sin(phase_diff), axis=1)
    
    # Update phases: dtheta/dt = omega + coupling
    new_phases = phases + dt * (omega + coupling)
    return new_phases % (2 * jnp.pi)  # Keep phases in [0, 2π]

def calculate_order_parameter(phases):
    """Calculate order parameter r"""
    z = jnp.mean(jnp.exp(1j * phases))  # Complex order parameter
    return jnp.abs(z), jnp.angle(z)  # Magnitude and average phase

# Parameters
N = 12  # Number of oscillators
dt = 0.04  # Time step
K_weak = 0.04   # Weak coupling (below critical)
K_strong = 25.0  # Strong coupling (above critical)

# Set up natural frequencies with some spread
key = random.PRNGKey(123)
omega_key, phase_key1, phase_key2 = random.split(key, 3)

fixed_random = False  # Set to True for some fixed random frequencies
frequency_spread = 2.5  # Spread of natural frequencies

if fixed_random:
    # random keys for different degrees of mixed frequencies
    key1, key2, key3 = random.split(omega_key, 3)

    # Create base frequency distribution
    omega = random.normal(key1, (N,)) * frequency_spread

    # Choose which oscillators will share a frequency (N//3 oscillators)
    group_indices = random.choice(key2, N, (N//3,), replace=False)

    # Generate a shared frequency for the group
    shared_freq = random.normal(key3, ()) * frequency_spread

    # Assign the shared frequency to group members
    omega = omega.at[group_indices].set(shared_freq)
else:
    # Natural frequencies with moderate spread
    omega = random.normal(omega_key, (N,)) * frequency_spread

# Initial phases
phases_weak = random.uniform(phase_key1, (N,), maxval=2*jnp.pi)
phases_strong = random.uniform(phase_key2, (N,), maxval=2*jnp.pi)

# Convert to numpy for matplotlib compatibility
phases_weak = np.array(phases_weak)
phases_strong = np.array(phases_strong)
omega = np.array(omega)

# Colors for each oscillator
colors = plt.cm.tab20(np.arange(N))

# Set up the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Animated Kuramoto Model: 6 Coupled Oscillators', fontsize=14, fontweight='bold')

# Set up both subplots
for ax, title in zip([ax1, ax2], [f'Weak Coupling (K = {K_weak})', f'Strong Coupling (K = {K_strong})']):
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('cos(θ)')
    ax.set_ylabel('sin(θ)')

# Unit circles
theta_circle = np.linspace(0, 2*np.pi, 100)
ax1.plot(np.cos(theta_circle), np.sin(theta_circle), 'lightgray', alpha=0.5, linewidth=1)
ax2.plot(np.cos(theta_circle), np.sin(theta_circle), 'lightgray', alpha=0.5, linewidth=1)

# Initialize oscillator points
oscillators1 = []
oscillators2 = []
labels1 = []
labels2 = []

for i in range(N):
    # Weak coupling oscillators
    osc1 = ax1.scatter([], [], s=150, color=colors[i], edgecolor='black', 
                      linewidth=1, zorder=10, label=f'Osc {i+1}')
    oscillators1.append(osc1)
    
    # Strong coupling oscillators  
    osc2 = ax2.scatter([], [], s=150, color=colors[i], edgecolor='black', 
                      linewidth=1, zorder=10)
    oscillators2.append(osc2)

# Order parameter arrows
arrow1 = ax1.annotate('', xy=(0, 0), xytext=(0, 0), 
                     arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.8))
arrow2 = ax2.annotate('', xy=(0, 0), xytext=(0, 0), 
                     arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.8))

# Order parameter text
r_text1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10, 
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
r_text2 = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=10, 
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Time text
time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=10)

def animate(frame):
    """Animation function"""
    global phases_weak, phases_strong
    
    # Update phases
    phases_weak = kuramoto_step(phases_weak, omega, K_weak, dt)
    phases_strong = kuramoto_step(phases_strong, omega, K_strong, dt)
    
    # Convert back to numpy
    phases_weak_np = np.array(phases_weak)
    phases_strong_np = np.array(phases_strong)
    
    # Calculate positions
    x_weak = np.cos(phases_weak_np)
    y_weak = np.sin(phases_weak_np)
    x_strong = np.cos(phases_strong_np)
    y_strong = np.sin(phases_strong_np)
    
    # Update oscillator positions
    for i in range(N):
        oscillators1[i].set_offsets([[x_weak[i], y_weak[i]]])
        oscillators2[i].set_offsets([[x_strong[i], y_strong[i]]])
    
    # Calculate and update order parameters
    r_weak, psi_weak = calculate_order_parameter(phases_weak_np)
    r_strong, psi_strong = calculate_order_parameter(phases_strong_np)
    
    # Update order parameter arrows
    arrow_x_weak = float(r_weak * np.cos(psi_weak))
    arrow_y_weak = float(r_weak * np.sin(psi_weak))
    arrow_x_strong = float(r_strong * np.cos(psi_strong))
    arrow_y_strong = float(r_strong * np.sin(psi_strong))
    
    arrow1.set_position((arrow_x_weak, arrow_y_weak))
    arrow1.xy = (arrow_x_weak, arrow_y_weak)
    arrow1.xytext = (0, 0)
    
    arrow2.set_position((arrow_x_strong, arrow_y_strong))
    arrow2.xy = (arrow_x_strong, arrow_y_strong)
    arrow2.xytext = (0, 0)
    
    # Update order parameter text
    r_text1.set_text(f'r = {float(r_weak):.3f}')
    r_text2.set_text(f'r = {float(r_strong):.3f}')
    
    # Update time
    time_text.set_text(f'Time: {frame * dt:.1f}s')
    
    return oscillators1 + oscillators2 + [arrow1, arrow2, r_text1, r_text2, time_text]

# Create animation
print("Creating animation...")
print("Close the plot window to stop the animation.")
print(f"Weak coupling K = {K_weak} (should show low synchronization)")
print(f"Strong coupling K = {K_strong} (should show high synchronization)")
print("\nWatch how:")
print("- Left panel: oscillators spread out, order parameter (red arrow) is small")
print("- Right panel: oscillators cluster together, order parameter is large")

# Add legend to first subplot
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

anim = animation.FuncAnimation(fig, animate, frames=2000, interval=100, blit=False, repeat=True)

# Add instructions
fig.text(0.5, 0.95, 'Red arrows show order parameter (synchronization strength)', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.show()

# Print final statistics
print(f"\nFinal synchronization measures:")
r_final_weak, _ = calculate_order_parameter(phases_weak)
r_final_strong, _ = calculate_order_parameter(phases_strong)
print(f"Weak coupling (K={K_weak}): r = {float(r_final_weak):.3f}")
print(f"Strong coupling (K={K_strong}): r = {float(r_final_strong):.3f}")
print(f"Synchronization improvement: {float(r_final_strong/r_final_weak):.1f}x")