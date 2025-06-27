# REU 2025 Summer -- Python Setup Instructions and General FAQs

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
