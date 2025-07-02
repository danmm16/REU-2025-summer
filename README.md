# REU 2025 Summer -- Python Setup Instructions and General FAQs

## July 2nd -- Kuramoto Model: Collective Synchronization Analysis
## July 1st -- Four-Mass System Normal Modes Analysis (similar to below)
## June 30th -- Three-Mass System Normal Modes Analysis

---

## July 2nd -- Kuramoto Model: Collective Synchronization Analysis

### FIRST: Get the Code Files

#### Option A: Download from Repository (Recommended)

1. **Go to the repository URL** (provided by your instructor or TA)
2. **Click the green "Code" button**
3. **Select "Download ZIP"**
4. **Extract/unzip the downloaded file** to a folder on your computer
5. **Navigate to the extracted folder** - you should see files like `kuramoto_6_bodies.py`

#### Option B: Clone with Git (If you have Git installed)

```bash
git clone https://github.com/danmm16/REU-2025-summer.git
cd path/to/your/REU-2025-summer/July\ 2/
```

#### Option C: Manual Download

If files are provided individually:
1. **Create a new folder** for this project (e.g., "kuramoto_analysis")
2. **Save each file** in this folder:
   - `kuramoto.py`
   - `bifurcation.py`
   - `kuramoto.md`

### NEXT: Get Required Python Packages

Before running the Kuramoto model code, you need to install the following packages:

#### Installation Commands

Open your command prompt/terminal and run these commands one by one:

```bash
pip install jax
pip install jaxlib
pip install matplotlib
pip install numpy
```

**Alternative (install all at once):**
```bash
pip install jax jaxlib matplotlib numpy
```

#### Package Purposes

- **JAX**: High-performance numerical computing with automatic differentiation
- **JAXlib**: JAX's supporting library for linear algebra operations
- **Matplotlib**: Creating plots and visualizations
- **NumPy**: Mathematical operations and arrays (used by JAX)

#### JAX Installation Notes

JAX provides significant performance improvements for numerical computations. If you encounter issues with JAX installation:

**Alternative (CPU-only JAX):**
```bash
pip install --upgrade "jax[cpu]"
```

**For GPU support (advanced users):**
```bash
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### THEN: File Organization

Create a new folder for this project and save these files:

1. `kuramoto.py` - Educational 6-oscillator visualization
2. `bifurcation.py` - Comprehensive bifurcation analysis
3. `kuramoto_animated.py` - 12-oscillator demonstraion, animated
4. `kuramoto.md` - Theoretical background and explanations

### FINALLY: (How to) Run the Code

#### Starting with the Basics
**Run the 6-oscillator version first and 12-oscillator animated version next:**
```bash
python kuramoto.py
```

```bash
python kuramoto_animated.py
```

**Then explore the bifurcation analysis:**
```bash
python bifurcation.py
```

#### Option 1: Command Line
1. Open command prompt/terminal
2. Navigate to your project folder: `cd path/to/your/folder`
3. Run: `python kuramoto_6_bodies.py`

#### Option 2: Python IDE (Recommended)
1. Open your Python IDE (IDLE, PyCharm, VS Code, etc.)
2. Open `kuramoto_6_bodies.py` first
3. Click "Run" or press F5

### POST: What You Should See

#### From `kuramoto_6_bodies.py`:
1. **Plot window** with two subplots:
   - Left: Non-synchronized oscillators randomly distributed on unit circle
   - Right: Synchronized oscillators clustered together
   - Each oscillator numbered and color-coded
   - Red arrows showing order parameter (synchronization strength)

2. **Console output** showing:
   - Individual oscillator phases
   - Order parameter values
   - Synchronization comparison

#### From `kuramoto_bifurcation.py`:
1. **Progress messages** in console during computation
2. **Comprehensive plot window** with four panels:
   - Main bifurcation diagram (order parameter vs coupling strength)
   - Critical region zoom
   - Phase snapshots at different coupling strengths
3. **Statistical summary** showing:
   - Theoretical vs empirical critical coupling
   - System parameters and final order parameters

### PHYSICS: Understanding the Kuramoto Model

#### What You'll Observe:

1. **6-Oscillator System**:
   - **Non-synchronized**: Oscillators spread randomly around circle (low order parameter)
   - **Synchronized**: Oscillators cluster together (high order parameter)
   - **Individual tracking**: Each oscillator labeled and color-coded

2. **Bifurcation Analysis**:
   - **Below K_c**: Incoherent state (r ≈ 0)
   - **At K_c**: Critical transition point where synchronization emerges
   - **Above K_c**: Synchronized state (r approaches 1)

#### Key Concepts:

- **Order Parameter (r)**: Measures synchronization strength (0 = no sync, 1 = perfect sync)
- **Coupling Strength (K)**: How strongly oscillators influence each other
- **Critical Coupling (K_c)**: Threshold where synchronization transition occurs
- **Phase Transition**: Sudden change from disorder to order
- **Natural Frequencies**: Individual oscillator frequencies without coupling

### ERRORS: Troubleshooting Common Issues

#### Problem: "ModuleNotFoundError: No module named 'jax'"
**Solution:** Install JAX:
```bash
pip install jax jaxlib
```

#### Problem: JAX installation fails
**Solutions:**
- Try CPU-only version: `pip install --upgrade "jax[cpu]"`
- On Windows, make sure you have Visual Studio Build Tools
- Try conda instead: `conda install -c conda-forge jax`

#### Problem: "jax.config.update" not working
**Solution:** Your JAX version might be old. Update it:
```bash
pip install --upgrade jax jaxlib
```

#### Problem: Plots appear but are empty/blank
**Solutions:**
- Check if matplotlib backend is properly configured
- Try adding `plt.ion()` at the beginning of the script
- Ensure you're not running in a headless environment

#### Problem: Bifurcation analysis runs very slowly
**Solutions:**
- JAX should provide significant speedup - ensure it's properly installed
- Reduce `n_realizations` from 20 to 10 for faster computation
- Reduce `K_values` array size for quicker exploration

#### Problem: "Permission denied" when installing packages
**Solutions:**
- Add `--user` flag: `pip install --user jax jaxlib matplotlib`
- On Mac/Linux, try `sudo pip install` (use with caution)
- Consider using virtual environments

### FORMAT: Understanding the Code Structure

#### `kuramoto_6_bodies.py` Components:

1. **Parameter Setup**: 
   - Number of oscillators (6)
   - Natural frequency distributions
   - Initial phase configurations

2. **Physics Calculation**:
   - Order parameter computation
   - Complex synchronization measure
   - Individual oscillator tracking

3. **Visualization**:
   - Unit circle representation
   - Color-coded oscillators
   - Order parameter vectors

#### `kuramoto_bifurcation.py` Components:

1. **Simulation Engine**:
   - `kuramoto_step()` - Single integration step
   - `simulate_kuramoto()` - Full system evolution
   - JAX optimization for performance

2. **Bifurcation Analysis**:
   - Systematic coupling strength variation
   - Multiple realizations for statistical averaging
   - Critical point identification

3. **Comprehensive Visualization**:
   - Main bifurcation diagram
   - Critical region zoom
   - Representative phase snapshots

#### Key Parameters You Can Modify:

```python
# In kuramoto_6_bodies.py:
n_oscillator = 6              # Number of oscillators
frequency_spread = 0.5        # Natural frequency diversity

# In kuramoto_bifurcation.py:
N = 50                        # Number of oscillators
n_realizations = 20           # Statistical averaging
K_values = np.linspace(0, 4, 80)  # Coupling strength range
```

### AFTERWARD: Learning Exercises

#### Beginner Level:
1. Run both scripts and observe the different visualizations
2. In 6-oscillator code, try changing `n_oscillator` to 3, 4, or 8
3. Compare synchronized vs non-synchronized cases

#### Intermediate Level:
1. Modify the frequency distributions in `kuramoto_6_bodies.py`:
   ```python
   # Try uniform distribution
   omega = np.random.uniform(-1, 1, N)
   
   # Try all identical frequencies
   omega = np.zeros(N)
   ```

2. In bifurcation analysis, change the number of oscillators:
   ```python
   N = 20  # Smaller system
   N = 100 # Larger system
   ```

3. Adjust the coupling strength range to focus on critical region:
   ```python
   K_values = np.linspace(0.5, 3.0, 100)
   ```

#### Advanced Level:
1. **Add damping** to the Kuramoto equation
2. **Implement different network topologies** (not all-to-all coupling)
3. **Study finite-size effects** by varying system size
4. **Create time-dependent coupling** K(t)
5. **Analyze correlation functions** and relaxation times

### PHYSICS INSIGHTS: What to Look For

#### System Behavior:
- **Frequency Diversity Effects**: How spread in natural frequencies affects critical coupling
- **Finite Size Effects**: How system size influences the transition sharpness
- **Statistical Fluctuations**: Variability in order parameter near critical point
- **Scaling Behavior**: How the transition width scales with system parameters

#### Experimental Questions:
1. What happens when all oscillators have identical natural frequencies?
2. How does the critical coupling change with frequency distribution width?
3. Can you identify the critical coupling from the steepest part of the bifurcation curve?
4. How do finite-size fluctuations affect the transition?

### TIPS: Getting the Most from the Simulation

#### Best Practices:
1. **Start with 6-oscillator visualization** to understand basic concepts
2. **Progress to bifurcation analysis** for quantitative understanding
3. **Compare theoretical predictions** with simulation results
4. **Explore parameter space systematically**
5. **Connect to real-world examples** (fireflies, brain networks, power grids)

#### Performance Tips:
1. **JAX provides significant speedup** - ensure proper installation
2. **Reduce realizations** for initial exploration, increase for final analysis
3. **Use appropriate time steps** for numerical stability
4. **Monitor convergence** of statistical averages

### (FOR FUTURE USE:) Notes for Instructor and TAs

#### Pre-Class Preparation:
- **Test JAX installation** on lab computers - can be tricky on some systems
- **Prepare parameter sets** that demonstrate key physics concepts
- **Have backup NumPy versions** ready if JAX installation fails
- **Review synchronization theory** and critical phenomena concepts

#### Teaching Strategies:
- **Start with 6-oscillator demo** - intuitive visualization of synchronization
- **Connect to everyday examples** - metronomes, fireflies, applause
- **Emphasize critical phenomena** - universal behavior near transitions
- **Show real-world applications** - power grids, neural networks, biological systems
- **Link to statistical physics** - order parameters, phase transitions

#### Common Student Difficulties:
- **JAX installation issues** - provide alternative NumPy implementations
- **Complex number interpretation** - explain order parameter calculation
- **Statistical averaging concept** - emphasize why multiple realizations are needed
- **Critical point identification** - help students recognize sharp transitions
- **Physics vs mathematics** - connect abstract model to real phenomena

#### Extension Activities:
- **Network topology variations** - implement small-world, scale-free networks
- **Second-order Kuramoto model** - include oscillator inertia
- **Experimental data analysis** - analyze real synchronization data
- **Chimera states** - explore partially synchronized states
- **Adaptive networks** - coupling strengths that evolve over time

#### Assessment Ideas:
- **Parameter prediction exercises** - "What happens if all frequencies are identical?"
- **Critical coupling estimation** - have students identify K_c from plots
- **Real-world connection tasks** - identify systems exhibiting Kuramoto-like behavior
- **Scaling analysis** - study how results change with system size
- **Numerical experiments** - design and execute parameter studies

#### Technical Notes:
- **JAX performance benefits** - significant speedup for large systems
- **Numerical precision** - JAX enables high-precision calculations
- **Random number generation** - JAX uses different RNG approach than NumPy
- **Compilation overhead** - first JAX run may be slower due to JIT compilation
- **Memory considerations** - large systems may require significant RAM

#### Pedagogical Connections:
- **Statistical physics**: Order parameters, phase transitions, critical phenomena
- **Complex systems**: Emergence, collective behavior, self-organization
- **Nonlinear dynamics**: Bifurcations, stability analysis, synchronization
- **Network science**: Coupled oscillators, complex networks, spreading phenomena
- **Applications**: Neuroscience, ecology, engineering, social systems

---

## July 1st -- Four-Mass System Normal Modes Analysis (similar to below)
## June 30th -- Three-Mass System Normal Modes Analysis

### FIRST: Get the Code Files

#### Option A: Download from Repository (Recommended)

1. **Go to the repository URL** (provided by your instructor or TA)
2. **Click the green "Code" button**
3. **Select "Download ZIP"**
4. **Extract/unzip the downloaded file** to a folder on your computer
5. **Navigate to the extracted folder** - you should see files like `three-spring-system.py`

#### Option B: Clone with Git (If you have Git installed)

```bash
git clone https://github.com/danmm16/REU-2025-summer.git
cd path/to/your/REU-2025-summer/June\ 30/
```

#### Option C: Manual Download

If files are provided individually:
1. **Create a new folder** for this project (e.g., "three_spring_analysis")
2. **Save each file** in this folder:
   - `three-spring-system.py`
   - `spring_system_config.txt` (optional configuration file)

### NEXT: Get Required Python Packages

Before running the three-spring system code, you need to install the following packages:

#### Installation Commands

Open your command prompt/terminal and run these commands one by one:

```bash
pip install numpy
pip install vpython
```

**Alternative (install all at once):**
```bash
pip install numpy vpython
```

#### Package Purposes

- **NumPy**: Mathematical operations, arrays, and eigenvalue analysis
- **VPython**: 3D visualization and interactive simulation

### CONFIGURATION: Setting Up Parameters

#### Create Configuration File (Optional)

Create a file named `spring_system_config.txt` in your project folder with the following content:

```
# Three-Spring System Configuration File
# Lines starting with # are comments and will be ignored
# Format: parameter = value

# Mass parameters (kg)
m1 = 0.15
m2 = 0.3
m3 = 0.15

# Spring constants (N/m)
k1 = 10.0
k2 = 5.0
k3 = 5.0
k4 = 10.0

# Initial conditions
# Position displacements (m)
x1_initial = 0.03
x2_initial = 0.0
x3_initial = 0.0

# Initial velocities (m/s)
x1dot_initial = 0.0
x2dot_initial = 0.0
x3dot_initial = 0.0

# Simulation parameters
dt = 0.01
amplitude = 0.05
initial_mode = 0

# Visual parameters
system_length = 0.5
mass_scale_factor = 0.1
```

#### Sample Configurations to Try

Add these to your config file to experiment with different behaviors:

```
# Configuration 1: Equal masses, different springs
# m1 = 0.2
# m2 = 0.2
# m3 = 0.2
# k1 = 15.0
# k2 = 3.0
# k3 = 8.0
# k4 = 12.0

# Configuration 2: Very different masses
# m1 = 0.05
# m2 = 0.5
# m3 = 0.1

# Configuration 3: Weak coupling
# k2 = 1.0
# k3 = 1.0
```

### THEN: File Organization

Create a new folder for this project and save these files:

1. `three-spring-system.py` - Main interactive simulation
2. `spring_system_config.txt` - Configuration file (optional)

### FINALLY: (How to) Run the Code

#### Option 1: Command Line
1. Open command prompt/terminal
2. Navigate to your project folder: `cd path/to/your/folder`
3. Run: `python three-spring-system.py`

#### Option 2: Python IDE (Recommended)
1. Open your Python IDE (IDLE, PyCharm, VS Code, etc.)
2. Open `three-spring-system.py`
3. Click "Run" or press F5

**Note:** VPython opens in a web browser window, so make sure you have a web browser available.

### POST: What You Should See

When you run `three-spring-system.py`, you should see:

1. **Console output** showing:
   - Configuration loading status
   - Normal mode frequencies and eigenvectors
   - Current parameter values
   - Keyboard control instructions

2. **Browser window** with:
   - 3D visualization of the spring-mass system
   - Two graphs: Position vs Time and Normal Mode Shapes
   - Real-time differential equations display
   - Interactive control instructions

3. **Interactive features**:
   - Mass boxes that scale with mass values
   - Springs that stretch and compress realistically
   - Real-time parameter displays

### CONTROLS: Interactive Operation

#### Keyboard Controls (Click on 3D view first to focus)

**Simulation Control:**
- `Space` - Pause/Resume simulation
- `n` - Next normal mode
- `p` - Previous normal mode
- `r` - Reset simulation

**Parameter Adjustment:**
- `1` / `Shift+1` - Increase/decrease m₁ by 0.01 kg
- `2` / `Shift+2` - Increase/decrease m₂ by 0.01 kg
- `3` / `Shift+3` - Increase/decrease m₃ by 0.01 kg
- `q` / `Q` - Increase/decrease k₁ by 0.5 N/m
- `w` / `W` - Increase/decrease k₂ by 0.5 N/m
- `e` / `E` - Increase/decrease k₃ by 0.5 N/m
- `f` / `F` - Increase/decrease k₄ by 0.5 N/m

**Configuration Management:**
- `s` - Save current parameters to config file
- `l` - Reload parameters from config file

### PHYSICS: Understanding Normal Modes

#### What You'll Observe:

1. **Mode 1 (Lowest Frequency)**: All masses move together in phase
2. **Mode 2 (Middle Frequency)**: Anti-symmetric motion with outer masses moving opposite to center
3. **Mode 3 (Highest Frequency)**: Complex high-frequency oscillations

#### Key Concepts:

- **Eigenfrequencies**: Natural oscillation frequencies of the system
- **Mode Shapes**: Characteristic patterns of motion for each mode
- **Coupling**: How spring constants affect mode frequencies
- **Mass Effects**: How mass ratios influence mode characteristics

### ERRORS: Troubleshooting Common Issues

#### Problem: "ModuleNotFoundError: No module named 'vpython'"
**Solution:** Install VPython:
```bash
pip install vpython
```

#### Problem: "ModuleNotFoundError: No module named 'numpy'"
**Solution:** Install NumPy:
```bash
pip install numpy
```

#### Problem: Browser window doesn't open
**Solutions:**
- Make sure you have a web browser installed
- Try manually opening the URL shown in the console
- Check firewall settings that might block local connections

#### Problem: 3D visualization appears blank
**Solutions:**
- Try refreshing the browser window
- Check JavaScript is enabled in your browser
- Try a different web browser

#### Problem: Keyboard controls don't work
**Solutions:**
- Click on the 3D visualization area to focus it
- Make sure you're pressing keys while the browser window is active
- Try clicking directly on the 3D scene

#### Problem: "Permission denied" when installing packages
**Solutions:**
- Add `--user` flag: `pip install --user numpy vpython`
- On Mac/Linux, try `sudo pip install` (use with caution)

### FORMAT: Understanding the Code Structure

#### Main Components:

1. **Configuration System**: 
   - `read_config_file()` - Loads parameters from text file
   - `save_current_config()` - Saves current state to file

2. **Physics Engine**:
   - `calculate_normal_modes()` - Eigenvalue analysis
   - Force calculations using spring equations
   - Numerical integration for motion

3. **Visualization**:
   - 3D spring-mass system with VPython
   - Real-time graphs and displays
   - Dynamic mass sizing

4. **Interaction System**:
   - Keyboard event handling
   - Real-time parameter adjustment
   - Mode switching

#### Key Parameters You Can Modify:

```python
# In config file or by keyboard:
m1, m2, m3        # Mass values (kg)
k1, k2, k3, k4    # Spring constants (N/m)
dt                # Time step for simulation
amplitude         # Amplitude for mode excitation
initial_mode      # Starting mode (0, 1, 2, or 3)
```

### AFTERWARD: Learning Exercises

#### Beginner Level:
1. Run the simulation and observe all three normal modes
2. Try the sample configurations by uncommenting them in the config file
3. Use keyboard controls to adjust one parameter and observe changes

#### Intermediate Level:
1. Create equal masses and observe how mode shapes change
2. Make one spring very weak (k₂ = 0.5) and see decoupling effects
3. Try extreme mass ratios (m₁ = 0.05, m₂ = 0.5, m₃ = 0.05)

#### Advanced Level:
1. Create custom configurations that produce specific mode behaviors
2. Analyze how coupling strength affects frequency separation
3. Study the relationship between mass ratios and mode localization

### PHYSICS INSIGHTS: What to Look For

#### System Behavior:
- **Strong Coupling**: When k₂ and k₃ are large, modes are well-mixed
- **Weak Coupling**: When k₂ and k₃ are small, masses oscillate more independently
- **Mass Effects**: Heavy masses tend to stay stationary in high-frequency modes
- **Frequency Scaling**: √(k/m) relationship visible in mode frequencies

#### Experimental Questions:
1. What happens when all masses are equal?
2. How does weak coupling affect normal modes?
3. Can you create modes where only one mass moves significantly?
4. How do the differential equations change as you adjust parameters?

### TIPS: Getting the Most from the Simulation

#### Best Practices:
1. **Start with default parameters** to understand normal behavior
2. **Change one parameter at a time** to see specific effects
3. **Use the save/load features** to preserve interesting configurations
4. **Watch the differential equations** update in real-time
5. **Pay attention to mass box sizes** - they provide visual mass feedback

#### Debugging Tips:
1. **Check the console output** for configuration loading messages
2. **Save your work** frequently using the 's' key
3. **Reset the simulation** with 'r' if things get chaotic
4. **Use the config file** for reproducible experiments

### (FOR FUTURE USE:) Notes for Instructor and TAs

#### Pre-Class Preparation:
- **Test VPython installation** on lab computers beforehand
- **Prepare sample config files** with interesting parameter sets for demonstrations
- **Have backup static plots** ready in case VPython has browser issues
- **Consider creating a shared folder** with pre-made configuration files for different experiments

#### Teaching Strategies:
- **Start with Mode 1 demonstration** - easiest to understand conceptually
- **Show parameter effects live** - adjust masses/springs while projecting to class
- **Use the differential equation display** to connect math theory with simulation
- **Emphasize the eigenvalue problem** - this connects to linear algebra concepts
- **Connect to real-world examples** - buildings, bridges, molecular vibrations

#### Common Student Difficulties:
- **Browser focus issues** - students often forget to click on 3D view before using keyboard
- **Parameter adjustment confusion** - demonstrate SHIFT key usage for decreasing values
- **Normal mode interpretation** - help students understand that any motion is a superposition
- **Configuration file syntax** - watch for missing equals signs or invalid parameter names
- **Physics vs. simulation** - distinguish between ideal mathematical model and real systems

#### Extension Activities:
- **Create custom configurations** that demonstrate specific physics principles
- **Compare with analytical solutions** for simple cases (equal masses, etc.)
- **Add damping analysis** - modify code to include energy dissipation
- **Explore resonance** - drive the system at different frequencies
- **Connect to Fourier analysis** - analyze the frequency content of general motion

#### Assessment Ideas:
- **Parameter prediction exercises** - "What happens if k₂ = 0?"
- **Mode shape sketching** - have students predict eigenvectors before running simulation
- **Configuration challenges** - "Create a system where only the center mass moves in Mode 1"
- **Real-world connections** - identify systems that exhibit similar normal mode behavior
- **Troubleshooting scenarios** - give students broken config files to debug

#### Technical Notes:
- **VPython browser compatibility** - works best with Chrome/Firefox, may have issues with Safari
- **Performance considerations** - simulation may slow down on older computers with high dt
- **File permissions** - students may need help with file paths and saving config files
- **Network restrictions** - some institutional firewalls may block VPython's local server
- **Backup plans** - have screenshots/videos ready if live demo fails

#### Pedagogical Connections:
- **Linear algebra**: Eigenvalue/eigenvector concepts in action
- **Differential equations**: Second-order coupled ODEs with real physical meaning
- **Fourier analysis**: Mode superposition and frequency domain concepts
- **Engineering applications**: Structural dynamics, mechanical vibrations
- **Advanced topics**: Preparation for quantum mechanics (particle in a box similarities)

---

## June 27th -- Duffing Oscillator Analysis

### FIRST: Get the Code Files

#### Option A: Download from Repository (Recommended)

1. **Go to the repository URL** (provided by your instructor or TA)
2. **Click the green "Code" button**
3. **Select "Download ZIP"**
4. **Extract/unzip the downloaded file** to a folder on your computer
5. **Navigate to the extracted folder** - you should see files like `duffing_ode.py`

#### Option B: Clone with Git (If you have Git installed)

```bash
git clone https://github.com/danmm16/REU-2025-summer.git
cd path/to/your/REU-2025-summer/June\ 27/
```

#### Option C: Manual Download

If files are provided individually:
1. **Create a new folder** for this project (e.g., "duffing_analysis")
2. **Save each file** in this folder:
   - `duffing_ode.py`
   - `duffing_solution.py` 
   - `duffing_ode.md`
   - `duffing_solution.md`

### NEXT: Get Required Python Packages

Before running the Duffing oscillator code, you need to install the following packages. These are not included with Python by default:

#### Installation Commands

Open your command prompt/terminal and run these commands one by one:

```bash
pip install numpy
pip install scipy
pip install matplotlib
```

**Alternative (install all at once):**
```bash
pip install numpy scipy matplotlib
```

#### Package Purposes

- **NumPy**: Mathematical operations and arrays
- **SciPy**: Scientific computing (we use it for solving differential equations)
- **Matplotlib**: Creating plots and graphs

### THEN: File Organization

Create a new folder for this project and save these files:

1. `duffing_ode.py` - Main analysis script (run this one)
2. `duffing_solution.py` - Complete detailed analysis
3. `duffing_ode.md` - Background theory and explanations
4. `duffing_solution.md` - Detailed mathematical documentation

### FINALLY: (How to) Run the Code

#### Option 1: Command Line
1. Open command prompt/terminal
2. Navigate to your project folder: `cd path/to/your/folder`
3. Run: `python duffing_ode.py`

#### Option 2: Python IDE (Recommended for beginners)
1. Open your Python IDE (IDLE, PyCharm, VS Code, etc.)
2. Open `duffing_ode.py`
3. Click "Run" or press F5

#### Option 3: Jupyter Notebook (If you have it)
1. Start Jupyter: `jupyter notebook`
2. Create new notebook
3. Copy and paste code sections as needed

### POST: What You Should See

When you run `duffing_ode.py`, you should see:

1. **Console output** showing:
   - "Solving for Linear (c=0)..."
   - "Solving for Weakly Nonlinear (c=0.1)..."
   - "Solving for Strongly Nonlinear (c=1.0)..."
   - Harmonic analysis results

2. **Two plot windows**:
   - First window: 6 subplots showing time series and power spectra
   - Second window: Phase portraits comparison

### ERRORS: Troubleshooting Common Issues

#### Problem: "ModuleNotFoundError"
**Solution:** Install the missing package using pip:
```bash
pip install [package_name]
```

#### Problem: "pip is not recognized"
**Solutions:**
- Try `python -m pip install numpy scipy matplotlib`
- On Mac/Linux, try `pip3` instead of `pip`
- Make sure Python is added to your system PATH

#### Problem: Plots don't appear
**Solutions:**
- Make sure you're not running in a text-only environment
- Try adding `plt.show()` at the end if using interactive mode
- Check if you have display capabilities (especially on remote servers)

#### Problem: "Permission denied" when installing packages
**Solutions:**
- Add `--user` flag: `pip install --user numpy scipy matplotlib`
- On Mac/Linux, try `sudo pip install` (use with caution)

### FORMAT: Understanding the Code Structure

#### Main Components:

1. **`duffing_ode()` function**: Defines the differential equation
2. **`solve_duffing()` function**: Solves the equation numerically
3. **`compute_power_spectrum()` function**: Analyzes frequency content
4. **`analyze_duffing_oscillator()` function**: Main analysis routine

#### Key Parameters You Can Modify:

```python
c_values = [0.0, 0.1, 1.0]  # Nonlinearity strengths
t_span = (0, 20)            # Time range
t_eval = np.linspace(0, 20, 4000)  # Time resolution
```

### AFTERWARD: Learning Exercises

#### Beginner Level:
1. Run the code as-is and observe the outputs
2. Change `c_values` to `[0.0, 0.05, 0.2]` and see how results change
3. Modify the time span to `(0, 10)` for shorter simulation

#### Intermediate Level:
1. Add a new nonlinearity value: `c_values = [0.0, 0.1, 0.5, 1.0]`
2. Change initial conditions in the `solve_duffing()` function
3. Modify plot colors and labels

#### Advanced Level:
1. Add damping to the equation
2. Implement different initial conditions
3. Create additional analysis plots

### TIPS: Getting Help

If you encounter issues:

1. **Read error messages carefully** - they usually tell you what's wrong
2. **Check package installation** - run `pip list` to see installed packages
3. **Python version compatibility** - this code works with Python 3.6+
4. **Online resources**:
   - Python documentation: https://docs.python.org/
   - NumPy documentation: https://numpy.org/doc/
   - SciPy documentation: https://scipy.org/
   - Matplotlib documentation: https://matplotlib.org/

### (FOR FUTURE USE:) Notes for Instructor and TAs

- Students should run `duffing_ode.py` first for the basic analysis
- `duffing_solution.py` provides more comprehensive analysis
- The `.md` files contain theoretical background and detailed explanations
- Consider demonstrating the installation process if students are unfamiliar with pip
- Jupyter notebooks might be easier for students to experiment with code modifications