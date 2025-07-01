from vpython import *
import numpy as np
import os

def read_config_file(filename="four_mass_system_config.txt"):
    """Read configuration parameters from a text file"""
    config = {}
    default_config = {
        'm1': 0.25, 'm2': 0.25, 'm3': 0.25, 'm4': 0.25,
        'k1': 6.0, 'k2': 6.0, 'k3': 6.0, 'k4': 6.0, 'k5': 6.0,
        'x1_initial': 0.0, 'x2_initial': 0.0, 'x3_initial': 0.0, 'x4_initial': 0.0,
        'x1dot_initial': 0.0, 'x2dot_initial': 0.0, 'x3dot_initial': 0.0, 'x4dot_initial': 0.0,
        'dt': 0.01, 'amplitude': 0.05, 'initial_mode': 0,
        'system_length': 0.6, 'mass_scale_factor': 0.1
    }
    
    # Start with default values
    config = default_config.copy()
    
    try:
        if os.path.exists(filename):
            print(f"📁 Reading configuration from: {filename}")
            with open(filename, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    # Skip comments and empty lines
                    if line.startswith('#') or not line:
                        continue
                    
                    # Parse parameter = value
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        try:
                            # Convert to appropriate type
                            if key in config:
                                if key == 'initial_mode':
                                    config[key] = int(float(value))
                                else:
                                    config[key] = float(value)
                                print(f"  ✓ {key} = {config[key]}")
                            else:
                                print(f"  ⚠️  Unknown parameter '{key}' on line {line_num}")
                        except ValueError:
                            print(f"  ❌ Invalid value for '{key}' on line {line_num}: {value}")
            print("📋 Configuration loaded successfully!")
        else:
            print(f"📁 Config file '{filename}' not found, using default values")
            print("💡 Create 'four_mass_system_config.txt' to customize parameters")
            
    except Exception as e:
        print(f"❌ Error reading config file: {e}")
        print("🔄 Using default configuration")
    
    return config

def save_current_config(config, filename="four_mass_system_config_current.txt"):
    """Save current parameters to a config file"""
    try:
        with open(filename, 'w') as file:
            file.write("# Four-Spring System - Current Configuration\n")
            file.write("# Generated automatically\n\n")
            
            file.write("# Mass parameters (kg)\n")
            file.write(f"m1 = {config['m1']:.3f}\n")
            file.write(f"m2 = {config['m2']:.3f}\n")
            file.write(f"m3 = {config['m3']:.3f}\n")
            file.write(f"m4 = {config['m4']:.3f}\n\n")
            
            file.write("# Spring constants (N/m)\n")
            file.write(f"k1 = {config['k1']:.1f}\n")
            file.write(f"k2 = {config['k2']:.1f}\n")
            file.write(f"k3 = {config['k3']:.1f}\n")
            file.write(f"k4 = {config['k4']:.1f}\n")
            file.write(f"k5 = {config['k5']:.1f}\n\n")
            
            file.write("# Initial conditions\n")
            file.write(f"x1_initial = {config['x1_initial']:.3f}\n")
            file.write(f"x2_initial = {config['x2_initial']:.3f}\n")
            file.write(f"x3_initial = {config['x3_initial']:.3f}\n")
            file.write(f"x4_initial = {config['x4_initial']:.3f}\n")
            file.write(f"x1dot_initial = {config['x1dot_initial']:.3f}\n")
            file.write(f"x2dot_initial = {config['x2dot_initial']:.3f}\n")
            file.write(f"x3dot_initial = {config['x3dot_initial']:.3f}\n")
            file.write(f"x4dot_initial = {config['x4dot_initial']:.3f}\n\n")
            
            file.write("# Simulation parameters\n")
            file.write(f"dt = {config['dt']:.3f}\n")
            file.write(f"amplitude = {config['amplitude']:.3f}\n")
            file.write(f"initial_mode = {config['initial_mode']}\n")
            
        print(f"💾 Current configuration saved to: {filename}")
    except Exception as e:
        print(f"❌ Error saving config: {e}")

# Load configuration from file
config = read_config_file()

# Create main scene with locked camera
scene = canvas(title='Interactive Four-Spring System - Normal Modes Analysis', 
               width=1200, height=800, background=color.black,
               userzoom=False, userspin=False, userpan=False,
               center=vector(0,0,0), range=0.35)

# Create graphs
g1 = graph(title="Position vs Time", xtitle="t (s)", ytitle="x (m)", 
           width=700, height=250, align='left')
f1 = gcurve(graph=g1, color=color.blue, label="x1")
f2 = gcurve(graph=g1, color=color.red, label="x2")
f3 = gcurve(graph=g1, color=color.purple, label="x3")
f4 = gcurve(graph=g1, color=color.orange, label="x4")

g2 = graph(title="Normal Mode Shapes", xtitle="Mass Number", ytitle="Amplitude", 
           width=700, height=250, align='left')
mode1_curve = gcurve(graph=g2, color=color.yellow, label="Mode 1")
mode2_curve = gcurve(graph=g2, color=color.green, label="Mode 2")
mode3_curve = gcurve(graph=g2, color=color.cyan, label="Mode 3")
mode4_curve = gcurve(graph=g2, color=color.magenta, label="Mode 4")

# System parameters (loaded from config file)
class SystemParams:
    def __init__(self, config):
        self.m1 = config['m1']
        self.m2 = config['m2']
        self.m3 = config['m3']
        self.m4 = config['m4']
        self.k1 = config['k1']
        self.k2 = config['k2']
        self.k3 = config['k3']
        self.k4 = config['k4']
        self.k5 = config['k5']
    
    def to_dict(self):
        return {
            'm1': self.m1, 'm2': self.m2, 'm3': self.m3, 'm4': self.m4,
            'k1': self.k1, 'k2': self.k2, 'k3': self.k3, 'k4': self.k4, 'k5': self.k5,
            'x1_initial': 0, 'x2_initial': 0, 'x3_initial': 0, 'x4_initial': 0,
            'x1dot_initial': 0, 'x2dot_initial': 0, 'x3dot_initial': 0, 'x4dot_initial': 0,
            'dt': config['dt'], 'amplitude': config['amplitude'], 
            'initial_mode': 0
        }

params = SystemParams(config)

# Length parameters (from config)
L = config['system_length']
dL = L/5  # Now divided by 5 for 4 masses

# Mass visualization scaling
mass_scale_factor = config['mass_scale_factor']

# Create system visualization
left = sphere(pos=vector(-L/2,0,0), radius=0.01, color=color.red)
right = sphere(pos=vector(L/2,0,0), radius=0.01, color=color.red)

def calculate_box_size(mass):
    """Calculate box size proportional to mass"""
    base_size = 0.02
    width = base_size + mass_scale_factor * mass
    height = base_size + mass_scale_factor * mass * 0.7
    depth = base_size + mass_scale_factor * mass * 0.7
    return vector(width, height, depth)

# Mass positions (boxes) - size will be proportional to mass
car1 = box(pos=vector(left.pos.x+dL,0,0), size=calculate_box_size(params.m1), color=color.yellow)
car2 = box(pos=vector(left.pos.x+2*dL,0,0), size=calculate_box_size(params.m2), color=color.cyan)
car3 = box(pos=vector(left.pos.x+3*dL,0,0), size=calculate_box_size(params.m3), color=color.magenta)
car4 = box(pos=vector(left.pos.x+4*dL,0,0), size=calculate_box_size(params.m4), color=color.orange)

# Springs
spring1 = helix(pos=left.pos, axis=car1.pos-left.pos, radius=0.005, thickness=0.003, color=color.white)
spring2 = helix(pos=car1.pos, axis=car2.pos-car1.pos, radius=0.005, thickness=0.003, color=color.white)
spring3 = helix(pos=car2.pos, axis=car3.pos-car2.pos, radius=0.005, thickness=0.003, color=color.white)
spring4 = helix(pos=car3.pos, axis=car4.pos-car3.pos, radius=0.005, thickness=0.003, color=color.white)
spring5 = helix(pos=car4.pos, axis=right.pos-car4.pos, radius=0.005, thickness=0.003, color=color.white)

# Parameter display labels
param_display = label(pos=vector(-L/2, 0.18, 0), text='', color=color.white, height=12, box=False)
equation_display = label(pos=vector(L/2, 0.18, 0), text='', color=color.cyan, height=16, box=False)

def update_parameter_display():
    """Update the parameter display"""
    param_text = f"""Parameters:
m₁ = {params.m1:.2f} kg    k₁ = {params.k1:.1f} N/m
m₂ = {params.m2:.2f} kg    k₂ = {params.k2:.1f} N/m  
m₃ = {params.m3:.2f} kg    k₃ = {params.k3:.1f} N/m
m₄ = {params.m4:.2f} kg    k₄ = {params.k4:.1f} N/m
                   k₅ = {params.k5:.1f} N/m"""
    param_display.text = param_text

def update_equation_display():
    """Update the differential equation display"""
    eq_text = f"""DIFFERENTIAL EQUATIONS:

m₁ẍ₁ = -k₁x₁ - k₂(x₁ - x₂)
m₂ẍ₂ = -k₂(x₂ - x₁) - k₃(x₂ - x₃)  
m₃ẍ₃ = -k₃(x₃ - x₂) - k₄(x₃ - x₄)
m₄ẍ₄ = -k₄(x₄ - x₃) - k₅x₄

MATRIX FORM: Mẍ + Kx = 0

M = [{params.m1:.2f}   0     0     0  ]
    [ 0   {params.m2:.2f}   0     0  ]
    [ 0     0   {params.m3:.2f}   0  ]
    [ 0     0     0   {params.m4:.2f}]

K = [{params.k1+params.k2:.1f}  {-params.k2:.1f}   0     0  ]
    [{-params.k2:.1f}  {params.k2+params.k3:.1f}  {-params.k3:.1f}   0  ]
    [ 0   {-params.k3:.1f}  {params.k3+params.k4:.1f}  {-params.k4:.1f}]
    [ 0     0   {-params.k4:.1f}  {params.k4+params.k5:.1f}]"""
    equation_display.text = eq_text

# Initialize displays
update_parameter_display()
update_equation_display()

def calculate_normal_modes():
    """Calculate normal modes using eigenvalue analysis"""
    print("Calculating normal modes...")
    
    # Mass matrix (diagonal)
    M = np.array([[params.m1, 0, 0, 0],
                  [0, params.m2, 0, 0],
                  [0, 0, params.m3, 0],
                  [0, 0, 0, params.m4]])
    
    # Stiffness matrix
    K = np.array([[params.k1 + params.k2, -params.k2, 0, 0],
                  [-params.k2, params.k2 + params.k3, -params.k3, 0],
                  [0, -params.k3, params.k3 + params.k4, -params.k4],
                  [0, 0, -params.k4, params.k4 + params.k5]])
    
    # Solve generalized eigenvalue problem: K * v = lambda * M * v
    M_inv = np.linalg.inv(M)
    A = M_inv @ K
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Sort by frequency (eigenvalues are omega^2)
    frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)  # Convert to Hz
    
    # Sort modes by frequency
    sorted_indices = np.argsort(frequencies)
    frequencies = frequencies[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Normalize eigenvectors
    for i in range(4):
        eigenvectors[:, i] = eigenvectors[:, i] / np.max(np.abs(eigenvectors[:, i]))
    
    print(f"Normal frequencies (Hz): {frequencies}")
    print("Mode shapes:")
    for i in range(4):
        print(f"Mode {i+1}: [{eigenvectors[0,i]:.3f}, {eigenvectors[1,i]:.3f}, {eigenvectors[2,i]:.3f}, {eigenvectors[3,i]:.3f}]")
    
    return frequencies, eigenvectors

def plot_mode_shapes(frequencies, eigenvectors):
    """Plot the normal mode shapes"""
    masses = [1, 2, 3, 4]  # Mass numbers for x-axis
    
    # Clear previous plots
    mode1_curve.delete()
    mode2_curve.delete()
    mode3_curve.delete()
    mode4_curve.delete()
    
    # Plot each mode
    for i, mass_num in enumerate(masses):
        mode1_curve.plot(mass_num, eigenvectors[i, 0])
        mode2_curve.plot(mass_num, eigenvectors[i, 1])
        mode3_curve.plot(mass_num, eigenvectors[i, 2])
        mode4_curve.plot(mass_num, eigenvectors[i, 3])

def set_initial_conditions_for_mode(mode_number, amplitude, frequencies, eigenvectors):
    """Set initial conditions to excite a specific normal mode"""
    mode_vector = eigenvectors[:, mode_number]
    
    # Set positions according to mode shape
    x1 = amplitude * mode_vector[0]
    x2 = amplitude * mode_vector[1]
    x3 = amplitude * mode_vector[2]
    x4 = amplitude * mode_vector[3]
    
    # Set velocities to zero (starting from maximum displacement)
    x1dot = 0
    x2dot = 0
    x3dot = 0
    x4dot = 0
    
    return x1, x2, x3, x4, x1dot, x2dot, x3dot, x4dot

# Calculate initial normal modes
frequencies, eigenvectors = calculate_normal_modes()
plot_mode_shapes(frequencies, eigenvectors)

# Create control display
scene.caption = """
<div style='font-family: Arial; background: #1a1a1a; color: white; padding: 15px; border-radius: 10px;'>
<h3>🎛️ Four-Mass System: Configuration File + Keyboard Controls</h3>

<div style='margin-bottom: 15px; padding: 10px; background: #2a2a2a; border-radius: 8px;'>
<b>📁 Configuration Files:</b><br>
Create <code>four_mass_system_config.txt</code> to set initial parameters | 
<code>s</code> = Save current settings | <code>l</code> = Reload from file
</div>

<table style='width: 100%; color: white; font-size: 13px;'>
<tr>
<td style='width: 25%; vertical-align: top;'>
<b>🎮 Simulation Control:</b><br>
<code>Space</code> - Pause/Resume<br>
<code>n</code> - Next Mode<br>
<code>p</code> - Previous Mode<br>
<code>r</code> - Reset Simulation<br>
<code>s</code> - Save Config<br>
<code>l</code> - Load Config<br>
</td>
<td style='width: 25%; vertical-align: top;'>
<b>⚖️ Mass Adjustment:</b><br>
<code>1</code> / <code>!</code> - m₁ +/- 0.01<br>
<code>2</code> / <code>@</code> - m₂ +/- 0.01<br>
<code>3</code> / <code>#</code> - m₃ +/- 0.01<br>
<code>4</code> / <code>$</code> - m₄ +/- 0.01<br>
</td>
<td style='width: 25%; vertical-align: top;'>
<b>🌀 Spring Constants:</b><br>
<code>q</code> / <code>Q</code> - k₁ +/- 0.5<br>
<code>w</code> / <code>W</code> - k₂ +/- 0.5<br>
<code>e</code> / <code>E</code> - k₃ +/- 0.5<br>
<code>f</code> / <code>F</code> - k₄ +/- 0.5<br>
<code>t</code> / <code>T</code> - k₅ +/- 0.5<br>
</td>
<td style='width: 25%; vertical-align: top;'>
<b>🎯 4 Normal Modes:</b><br>
Mode 1: In-phase motion<br>
Mode 2: Anti-symmetric<br>
Mode 3: Complex coupling<br>
Mode 4: High-frequency<br>
</td>
</tr>
</table>

<div style='margin-top: 15px; padding: 10px; background: #2a2a2a; border-radius: 8px;'>
<b>💡 Tips:</b> Use SHIFT + key to decrease values | Click on the 3D view to focus for keyboard input | 
Mass box sizes change with mass values | Now with 4 normal modes and richer dynamics!
</div>
</div>
"""

# Simulation variables (from config)
mode_to_run = config['initial_mode']
amplitude = config['amplitude']
running = True
t = 0
dt = config['dt']

# Initial conditions (from config file)
x1, x2, x3, x4 = config['x1_initial'], config['x2_initial'], config['x3_initial'], config['x4_initial']
x1dot, x2dot, x3dot, x4dot = config['x1dot_initial'], config['x2dot_initial'], config['x3dot_initial'], config['x4dot_initial']

def update_mass_visualization():
    """Update the visual size of masses based on their values"""
    car1.size = calculate_box_size(params.m1)
    car2.size = calculate_box_size(params.m2)
    car3.size = calculate_box_size(params.m3)
    car4.size = calculate_box_size(params.m4)

def update_system():
    """Recalculate normal modes and update displays when parameters change"""
    global frequencies, eigenvectors
    frequencies, eigenvectors = calculate_normal_modes()
    plot_mode_shapes(frequencies, eigenvectors)
    update_parameter_display()
    update_equation_display()
    update_mass_visualization()  # Update visual mass sizes
    
    # Update mode text
    if mode_to_run < 4:
        mode_text.text = f'Mode {mode_to_run + 1} ({frequencies[mode_to_run]:.3f} Hz)'
        mode_text.color = [color.yellow, color.green, color.cyan, color.magenta][mode_to_run]
    else:
        mode_text.text = 'Original Motion'
        mode_text.color = color.red

def switch_mode(new_mode):
    """Switch to a different mode"""
    global mode_to_run, x1, x2, x3, x4, x1dot, x2dot, x3dot, x4dot, t
    mode_to_run = new_mode
    t = 0  # Reset time
    
    if mode_to_run < 4:  # Normal modes
        x1, x2, x3, x4, x1dot, x2dot, x3dot, x4dot = set_initial_conditions_for_mode(
            mode_to_run, amplitude, frequencies, eigenvectors)
        mode_text.text = f'Mode {mode_to_run + 1} ({frequencies[mode_to_run]:.3f} Hz)'
        mode_text.color = [color.yellow, color.green, color.cyan, color.magenta][mode_to_run]
    else:  # Original motion
        x1, x2, x3, x4 = 0.03, 0, 0, 0
        x1dot, x2dot, x3dot, x4dot = 0, 0, 0, 0
        mode_text.text = 'Original Motion'
        mode_text.color = color.red
    
    # Clear graphs
    f1.delete()
    f2.delete()
    f3.delete()
    f4.delete()

# Apply initial mode if specified in config
if config['initial_mode'] < 4:
    x1, x2, x3, x4, x1dot, x2dot, x3dot, x4dot = set_initial_conditions_for_mode(
        mode_to_run, amplitude, frequencies, eigenvectors)

# Mode display
if mode_to_run < 4:
    mode_text = label(pos=vector(0, -0.15, 0), text=f'Mode {mode_to_run + 1} ({frequencies[mode_to_run]:.3f} Hz)', 
                      color=[color.yellow, color.green, color.cyan, color.magenta][mode_to_run], height=15, box=False)
else:
    mode_text = label(pos=vector(0, -0.15, 0), text='Original Motion', 
                      color=color.red, height=15, box=False)

def keydown(evt):
    global running, mode_to_run
    
    if evt.key == ' ':
        running = not running
        print("Paused" if not running else "Resumed")
    
    elif evt.key == 'n':
        mode_to_run = (mode_to_run + 1) % 5  # Now 5 modes (4 normal + original)
        switch_mode(mode_to_run)
        print(f"Switched to mode {mode_to_run + 1 if mode_to_run < 4 else 'Original'}")
    
    elif evt.key == 'p':
        mode_to_run = (mode_to_run - 1) % 5
        switch_mode(mode_to_run)
        print(f"Switched to mode {mode_to_run + 1 if mode_to_run < 4 else 'Original'}")
    
    elif evt.key == 'r':
        switch_mode(mode_to_run)  # Reset current mode
        print("Reset simulation")
    
    elif evt.key == 's':
        # Save current configuration
        current_config = params.to_dict()
        current_config.update({
            'x1_initial': x1, 'x2_initial': x2, 'x3_initial': x3, 'x4_initial': x4,
            'x1dot_initial': x1dot, 'x2dot_initial': x2dot, 'x3dot_initial': x3dot, 'x4dot_initial': x4dot,
            'dt': dt, 'amplitude': amplitude, 'initial_mode': mode_to_run
        })
        save_current_config(current_config)
    
    elif evt.key == 'l':
        # Reload configuration from file
        print("🔄 Reloading configuration...")
        new_config = read_config_file()
        params.m1, params.m2, params.m3, params.m4 = new_config['m1'], new_config['m2'], new_config['m3'], new_config['m4']
        params.k1, params.k2, params.k3, params.k4, params.k5 = new_config['k1'], new_config['k2'], new_config['k3'], new_config['k4'], new_config['k5']
        update_system()
        print("✅ Configuration reloaded!")
    
    # Mass parameter adjustments
    elif evt.key == '1':
        params.m1 += 0.01
        update_system()
        print(f"m1 = {params.m1:.2f}")
    elif evt.key == '!':
        params.m1 = max(0.01, params.m1 - 0.01)
        update_system()
        print(f"m1 = {params.m1:.2f}")
    
    elif evt.key == '2':
        params.m2 += 0.01
        update_system()
        print(f"m2 = {params.m2:.2f}")
    elif evt.key == '@':
        params.m2 = max(0.01, params.m2 - 0.01)
        update_system()
        print(f"m2 = {params.m2:.2f}")
    
    elif evt.key == '3':
        params.m3 += 0.01
        update_system()
        print(f"m3 = {params.m3:.2f}")
    elif evt.key == '#':
        params.m3 = max(0.01, params.m3 - 0.01)
        update_system()
        print(f"m3 = {params.m3:.2f}")
    
    elif evt.key == '4':
        params.m4 += 0.01
        update_system()
        print(f"m4 = {params.m4:.2f}")
    elif evt.key == '$':
        params.m4 = max(0.01, params.m4 - 0.01)
        update_system()
        print(f"m4 = {params.m4:.2f}")
    
    # Spring constant adjustments
    elif evt.key == 'q':
        params.k1 += 0.5
        update_system()
        print(f"k1 = {params.k1:.1f}")
    elif evt.key == 'Q':
        params.k1 = max(0.1, params.k1 - 0.5)
        update_system()
        print(f"k1 = {params.k1:.1f}")
    
    elif evt.key == 'w':
        params.k2 += 0.5
        update_system()
        print(f"k2 = {params.k2:.1f}")
    elif evt.key == 'W':
        params.k2 = max(0.1, params.k2 - 0.5)
        update_system()
        print(f"k2 = {params.k2:.1f}")
    
    elif evt.key == 'e':
        params.k3 += 0.5
        update_system()
        print(f"k3 = {params.k3:.1f}")
    elif evt.key == 'E':
        params.k3 = max(0.1, params.k3 - 0.5)
        update_system()
        print(f"k3 = {params.k3:.1f}")
    
    # Note: 'r' is used for reset, so k4 uses different keys
    elif evt.key == 'u':  # Changed from 'r' to 'u'
        params.k4 += 0.5
        update_system()
        print(f"k4 = {params.k4:.1f}")
    elif evt.key == 'U':
        params.k4 = max(0.1, params.k4 - 0.5)
        update_system()
        print(f"k4 = {params.k4:.1f}")
    
    elif evt.key == 't':
        params.k5 += 0.5
        update_system()
        print(f"k5 = {params.k5:.1f}")
    elif evt.key == 'T':
        params.k5 = max(0.1, params.k5 - 0.5)
        update_system()
        print(f"k5 = {params.k5:.1f}")

scene.bind('keydown', keydown)

print("🎛️ Interactive Four-Spring System with Configuration Files")
print("=" * 70)
print("📁 CONFIGURATION SYSTEM:")
print("   • Create 'four_mass_system_config.txt' to customize initial parameters")
print("   • Press [s] to save current settings")
print("   • Press [l] to reload from config file")
print("\n📋 KEYBOARD CONTROLS:")
print("   Simulation: [Space] = Pause/Resume  [n] = Next Mode  [p] = Previous  [r] = Reset")
print("   Config:     [s] = Save Config      [l] = Load Config")
print("   Masses:     [1]/[!] = m₁ ±0.01     [2]/[@] = m₂ ±0.01     [3]/[#] = m₃ ±0.01     [4]/[$] = m₄ ±0.01")
print("   Springs:    [q]/[Q] = k₁ ±0.5      [w]/[W] = k₂ ±0.5      [e]/[E] = k₃ ±0.5")
print("               [u]/[U] = k₄ ±0.5      [t]/[T] = k₅ ±0.5")
print("   💡 Tip: Use SHIFT + key to DECREASE values, normal key to INCREASE")
print("\n🎯 LOADED CONFIGURATION:")
print(f"   m₁={params.m1:.3f}kg  m₂={params.m2:.3f}kg  m₃={params.m3:.3f}kg  m₄={params.m4:.3f}kg")
print(f"   k₁={params.k1:.1f}N/m  k₂={params.k2:.1f}N/m  k₃={params.k3:.1f}N/m  k₄={params.k4:.1f}N/m  k₅={params.k5:.1f}N/m")
print(f"   Initial: x₁={config['x1_initial']:.3f}m  x₂={config['x2_initial']:.3f}m  x₃={config['x3_initial']:.3f}m  x₄={config['x4_initial']:.3f}m")
print(f"\n🌊 FOUR NORMAL MODES: Now with {len(frequencies)} distinct normal modes!")
for i, freq in enumerate(frequencies):
    print(f"   Mode {i+1}: {freq:.3f} Hz")
print("\n👆 Click on the 3D view to focus, then use keyboard controls!")
print("=" * 70)

# Main simulation loop
while t < 200:
    if running:
        rate(100)
        
        # Calculate forces using current parameters
        F1 = -params.k1*x1 - params.k2*(x1-x2)
        F2 = -params.k2*(x2-x1) - params.k3*(x2-x3)
        F3 = -params.k3*(x3-x2) - params.k4*(x3-x4)
        F4 = -params.k4*(x4-x3) - params.k5*x4
        
        # Calculate accelerations
        x1ddot = F1/params.m1
        x2ddot = F2/params.m2
        x3ddot = F3/params.m3
        x4ddot = F4/params.m4
        
        # Update velocities
        x1dot = x1dot + x1ddot*dt
        x2dot = x2dot + x2ddot*dt
        x3dot = x3dot + x3ddot*dt
        x4dot = x4dot + x4ddot*dt
        
        # Update positions
        x1 = x1 + x1dot*dt
        x2 = x2 + x2dot*dt
        x3 = x3 + x3dot*dt
        x4 = x4 + x4dot*dt
        
        # Update visual positions
        car1.pos = vector(left.pos.x+dL+x1, 0, 0)
        car2.pos = vector(left.pos.x+2*dL+x2, 0, 0)
        car3.pos = vector(left.pos.x+3*dL+x3, 0, 0)
        car4.pos = vector(left.pos.x+4*dL+x4, 0, 0)
        
        # Update springs
        spring1.axis = car1.pos - left.pos
        spring2.pos = car1.pos
        spring2.axis = car2.pos - car1.pos
        spring3.pos = car2.pos
        spring3.axis = car3.pos - car2.pos
        spring4.pos = car3.pos
        spring4.axis = car4.pos - car3.pos
        spring5.pos = car4.pos
        spring5.axis = right.pos - car4.pos
        
        # Update time
        t = t + dt
        
        # Plot data
        f1.plot(t, x1)
        f2.plot(t, x2)
        f3.plot(t, x3)
        f4.plot(t, x4)
    
    else:
        rate(30)  # Lower rate when paused

print("Simulation complete!")