import jax; jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Kuramoto model visualization: Non-synchronized vs Synchronized Oscillators
n_oscillator = 6

# Unit circle for reference
_thetas = jnp.arange(0, 2*jnp.pi, 0.01)

# Non-synchronized oscillators (random phases)
thetas_nonsync = random.uniform(random.PRNGKey(0), (n_oscillator,), maxval=2*jnp.pi)

# Synchronized oscillators (clustered around π/3)
thetas_sync = random.normal(random.PRNGKey(1), (n_oscillator,)) * 0.1 + jnp.pi / 3

# Calculate order parameter (synchronization measure) for non-synchronized
rx_nonsync = jnp.mean(jnp.cos(thetas_nonsync))
ry_nonsync = jnp.mean(jnp.sin(thetas_nonsync))
r_nonsync = jnp.sqrt(rx_nonsync**2 + ry_nonsync**2)

# Calculate order parameter for synchronized
rx_sync = jnp.mean(jnp.cos(thetas_sync))
ry_sync = jnp.mean(jnp.sin(thetas_sync))
r_sync = jnp.sqrt(rx_sync**2 + ry_sync**2)

# Create visualization
plt.figure(figsize=(12, 5))

# Non-synchronized oscillators plot
plt.subplot(1, 2, 1)
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.gca().set_aspect('equal')

# Unit circle
plt.plot(jnp.cos(_thetas), jnp.sin(_thetas), color="lightgray", ls="--", alpha=0.7, zorder=0)

# Individual oscillators
colors = plt.cm.tab10(jnp.arange(n_oscillator))
for i in range(n_oscillator):
    x, y = jnp.cos(thetas_nonsync[i]), jnp.sin(thetas_nonsync[i])
    plt.scatter(x, y, s=100, color=colors[i], zorder=10, edgecolor='black', linewidth=1)
    plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=8, ha='left')

# Order parameter vector
plt.arrow(0, 0, rx_nonsync, ry_nonsync, head_width=0.05, head_length=0.05, 
          fc='red', ec='red', linewidth=2, zorder=20)
plt.scatter([rx_nonsync], [ry_nonsync], color="red", s=50, zorder=30, marker='x')

plt.title(f"Non-synchronized Oscillators\nOrder parameter r = {r_nonsync:.3f}")
plt.xlabel("cos(θ)")
plt.ylabel("sin(θ)")
plt.grid(True, alpha=0.3)

# Synchronized oscillators plot
plt.subplot(1, 2, 2)
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.gca().set_aspect('equal')

# Unit circle
plt.plot(jnp.cos(_thetas), jnp.sin(_thetas), color="lightgray", ls="--", alpha=0.7, zorder=0)

# Individual oscillators
for i in range(n_oscillator):
    x, y = jnp.cos(thetas_sync[i]), jnp.sin(thetas_sync[i])
    plt.scatter(x, y, s=100, color=colors[i], zorder=10, edgecolor='black', linewidth=1)
    plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=8, ha='left')

# Order parameter vector
plt.arrow(0, 0, rx_sync, ry_sync, head_width=0.05, head_length=0.05, 
          fc='red', ec='red', linewidth=2, zorder=20)
plt.scatter([rx_sync], [ry_sync], color="red", s=50, zorder=30, marker='x')

plt.title(f"Synchronized Oscillators\nOrder parameter r = {r_sync:.3f}")
plt.xlabel("cos(θ)")
plt.ylabel("sin(θ)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print individual phases for reference
print("Non-synchronized phases (radians):")
for i, theta in enumerate(thetas_nonsync):
    print(f"Oscillator {i+1}: {theta:.3f}")

print("\nSynchronized phases (radians):")
for i, theta in enumerate(thetas_sync):
    print(f"Oscillator {i+1}: {theta:.3f}")