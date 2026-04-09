# BoundWalk — 1D Bounded Random Walk Simulator

A Python-based statistical mechanics simulator for studying thermalization, entropy production, and energy conservation in confined random walks.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

**BoundWalk** simulates a particle performing a random walk in a 1D bounded domain $[a, b]$ with reflecting (Neumann) boundary conditions. The simulator tracks statistical observables including Shannon entropy, KL divergence, and energy distribution, making it a pedagogical tool for exploring concepts from statistical mechanics and information theory.

### Key Features

- **Multiple step size modes**: Fixed, uniform distribution, or normal distribution
- **Energy conservation**: Perfect NVE (microcanonical ensemble) demonstration
- **Information-theoretic observables**: Shannon entropy $H[X_n]$ and KL divergence $D_{KL}(\mathbb{P} \| U)$
- **Real-time visualization**: Animated matplotlib plots with 5 synchronized subplots
- **Configurable physics**: Adjustable mass, time scale, and boundary conditions
- **Reproducible**: Optional RNG seeding for deterministic simulations

---

## Physics Background

BoundWalk approximates several physical systems:

### Microcanonical Ensemble (NVE)
Fixed step size → constant kinetic energy → Dirac delta energy distribution
```python
walk = BoundWalk(total_steps=1000, step_size=0.1, time_scale=0.5)
```

### Canonical Ensemble (NVT, future)
Temperature-controlled Gaussian steps → Boltzmann energy distribution

### Observables Tracked

| Observable | Formula | Physical Meaning |
|------------|---------|------------------|
| Shannon Entropy | $H[X_n] = -\sum_i p_i \ln p_i$ | Uncertainty in particle position |
| Boltzmann Entropy | $S = k_B \ln W$ | Maximum entropy for $W$ microstates |
| KL Divergence | $D_{KL}(P \| U)$ | Distance from equilibrium (uniform) |
| Kinetic Energy | $E = \frac{p^2}{2m}$ | Conserved quantity in NVE ensemble |

---

## Installation

### Requirements

- Python ≥ 3.13 (uses modern type hints)
- NumPy
- pandas
- matplotlib
- scipy
- Jupyter (for notebook interface)

### Setup

```bash
# Clone repository
git clone https://github.com/eigenname/1D-Random-Walk-Simulation.git
cd 1D-Random-Walk-Simulation

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## Quick Start

### Basic Usage

```python
from boundwalk import BoundWalk as BW

# NVE ensemble: fixed step size
walk_fixed = BW(
    total_steps=1000,
    step_size=0.1,
    time_scale=0.5,
    boundaries=(0, 1),
    seed=42
)

# Visualize all observables
walk_fixed.__visualize__()
```

### Advanced: Uniform Distribution

```python
from bw_tools import Uniform

walk_uniform = BW(
    total_steps=1000,
    step_size=Uniform(a=0.05, b=0.2, bin_width=0.01),
    time_scale=1.0,
    seed=42
)

walk_uniform.__visualize__()
```

### Accessing Data

```python
# Simulation data stored as pandas DataFrame
df = walk_fixed.data

# Extract specific observables
positions = df['nth Position']
entropies = df['nth Entropy']
energies = df['nth Energy']

# Verify energy conservation (NVE)
print(f"Energy mean: {energies.mean():.10f}")
print(f"Energy std:  {energies.std():.10f}")
```

---

## API Reference

### `BoundWalk` Class

```python
@dataclass(kw_only=True)
class BoundWalk:
    total_steps: int                      # Number of steps N
    step_size: float | int | dict         # Fixed or sampled (use Uniform/Normal)
    time_scale: float = 1.0               # Δt for velocity calculation
    boundaries: tuple = (0, 1)            # Reflecting domain [a, b]
    mass: float = 1.0                     # Particle mass
    initial_position: float = 0.0         # Starting position X₀
    initial_velocity: float = 0.0         # Starting velocity v₀
    seed: int = None                      # RNG seed (None = random)
    ms_between_frames: int = 500          # Animation frame interval
```

#### Data Attributes (after `__post_init__`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `self.displacements` | `np.ndarray` | Raw displacement array |
| `self.bin_width` | `float` | Histogram bin width |
| `self.domain` | `np.ndarray` | Domain grid $[a, b]$ |
| `self.data` | `pd.DataFrame` | Full simulation data |

#### DataFrame Columns

- `t`: Time (step index × time_scale)
- `n ≤ N`: Step index
- `nth Position`: Position $X_n$
- `nth Displacement`: Actual displacement (after reflection)
- `nth Velocity`: Velocity $v_n = \Delta x_n / \sqrt{\Delta t}$
- `nth Momentum`: Momentum $p_n = m v_n$
- `nth Energy`: Kinetic energy $E_n = p_n^2 / (2m)$
- `nth Possible Positions`: Array of visited positions
- `nth Position Probabilities`: Empirical distribution $P(X_n)$
- `nth Entropy`: Shannon entropy $H[X_n]$
- `nth KLD`: KL divergence $D_{KL}(\mathbb{P} \| U)$

### Factory Functions (`bw_tools.py`)

#### `Uniform(a, b, bin_width)`
Creates uniform step size distribution $\mathcal{U}[a, b]$.

```python
step_size = Uniform(a=0.1, b=0.5, bin_width=0.01)
```

#### `Normal(mean, std_dev, bin_width)`
Creates normal step size distribution $\mathcal{N}(\mu, \sigma^2)$.

```python
step_size = Normal(mean=0.0, std_dev=0.2, bin_width=0.01)
```

---

## Visualization Layout

```
┌─────────────────────┬─────────────────────┐
│  1D Position (live) │  P(Xₙ) Histogram    │
├─────────────────────┴─────────────────────┤
│           Position Xₙ vs t → N            │
├────────────────────────────────────────────┤
│           Entropy H[Xₙ] vs t → N          │
├────────────────────────────────────────────┤
│         KL Divergence D_KL vs t → N       │
├────────────────────────────────────────────┤
│         Energy Distribution P(E)          │
└────────────────────────────────────────────┘
```

**Plot Descriptions:**

1. **1D Position**: Real-time particle location on the number line
2. **Histogram**: Empirical position distribution converging to uniform
3. **Position vs t**: Trajectory showing confinement and reflections
4. **Entropy vs t**: Information growth toward Boltzmann limit $\ln W$
5. **KLD vs t**: Divergence from equilibrium decaying to zero
6. **Energy Distribution**: Dirac delta (NVE) or thermal distribution (NVT)

---

## Examples

### Example 1: Energy Conservation (NVE)

```python
from boundwalk import BoundWalk as BW

# Fixed step → constant energy
walk = BW(total_steps=1000, step_size=0.1, time_scale=0.5, seed=42)

# Verify conservation
energies = walk.data['nth Energy'].iloc[1:]  # Skip initial E=0
print(f"Unique energy values: {len(energies.unique())}")  # Should be 1
print(f"Energy = {energies.iloc[0]:.10f}")  # Constant value
```

**Expected Output:**
```
Unique energy values: 1
Energy = 0.0100000000
```

### Example 2: Thermalization Dynamics

```python
from bw_tools import Uniform

# Start particle at boundary
walk = BW(
    total_steps=5000,
    step_size=Uniform(0.01, 0.1, bin_width=0.01),
    initial_position=0.0,  # Left boundary
    boundaries=(0, 1),
    seed=42
)

# Track entropy approach to equilibrium
H_final = walk.data['nth Entropy'].iloc[-1]
H_max = np.log(len(walk.domain))
print(f"Final entropy: {H_final:.4f}")
print(f"Max entropy:   {H_max:.4f}")
print(f"Ratio:         {H_final/H_max:.2%}")
```

### Example 3: Comparing Step Size Distributions

```python
from bw_tools import Uniform, Normal
import matplotlib.pyplot as plt

walks = {
    'Fixed': BW(total_steps=1000, step_size=0.1, seed=42),
    'Uniform': BW(total_steps=1000, step_size=Uniform(0.05, 0.15, 0.01), seed=42),
    'Normal': BW(total_steps=1000, step_size=Normal(0.0, 0.1, 0.01), seed=42)
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, walk) in zip(axes, walks.items()):
    energies = walk.data['nth Energy'].iloc[1:]
    ax.hist(energies, bins=50, alpha=0.7, edgecolor='black')
    ax.set_title(f'{name} Step Size')
    ax.set_xlabel('Energy')
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()
```

---

## Development Roadmap

### Current: Animation 2.1 ✅
- [x] Fixed, uniform, and normal step size distributions
- [x] Energy tracking and conservation
- [x] 5-panel animated visualization
- [x] Velocity calculation fix for time_scale consistency

### Planned: Animation 2.2
- [ ] Phase space plot $(x, v)$ with energy contours
- [ ] Velocity autocorrelation function $\langle v(0) v(t) \rangle$
- [ ] Mean squared displacement (MSD) analysis
- [ ] Kinetic temperature $T_{kin} = m\langle v^2 \rangle / k_B$

### Future: Animation 3.0
- [ ] Langevin dynamics with friction: $\dot{v} = -\gamma v + \sigma \xi(t)$
- [ ] Explicit coupling to thermal bath
- [ ] Fluctuation-dissipation theorem verification
- [ ] 2D extension with periodic boundaries

---

## Technical Details

### Boundary Conditions

**Reflecting (Neumann) boundaries** via fold-back algorithm:
```python
def reflect(pos, left_bound, right_bound):
    while pos < left_bound or pos > right_bound:
        if pos < left_bound:
            pos = 2 * left_bound - pos
        elif pos > right_bound:
            pos = 2 * right_bound - pos
    return pos
```

Handles arbitrary step sizes exceeding domain width.

### Velocity-Energy Relationship

For time-discretized random walk with scaling parameter `time_scale`:

$$v_n = \frac{\Delta x_n}{\sqrt{\Delta t}}, \quad E_n = \frac{p_n^2}{2m} = \frac{m v_n^2}{2}$$

This ensures dimensional consistency: if displacements scale as $\Delta x \sim \sqrt{\Delta t}$, then $v \sim \Delta x / \sqrt{\Delta t}$ is independent of $\Delta t$.

### Numerical Precision

Rounding precision derived from `step_size` parameter:
- `step_size = 0.1` → 1 decimal place
- `step_size = 0.01` → 2 decimal places

Prevents floating-point accumulation errors in position tracking.

---

## Repository Structure

```
1D-Random-Walk-Simulation/
├── boundwalk.py              # BoundWalk class definition
├── bw_tools.py               # Helper functions and factories
├── BW_Animated_2_1.ipynb     # Jupyter notebook with examples
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes __pycache__/
└── LICENSE                   # MIT License
```

---

## Contributing

Contributions welcome! Areas of interest:

- **Performance**: Vectorization, JIT compilation (Numba)
- **Features**: New boundary conditions, multi-particle systems
- **Documentation**: Additional examples, physics derivations
- **Testing**: Unit tests for edge cases

Please open an issue or pull request on GitHub.

---

## Citation

If you use BoundWalk in academic work, please cite:

```bibtex
@software{boundwalk2025,
  author = {Hyunje Jun},
  title = {BoundWalk: 1D Bounded Random Walk Simulator},
  year = {2025},
  url = {https://github.com/eigenname/1D-Random-Walk-Simulation}
}
```

---

## License

MIT License - see `LICENSE` file for details.

---

## Acknowledgments

Developed as an extension of coursework in **PHSX 671: Thermal Physics** at the University of Kansas. Inspired by classical texts:
- Reif, *Fundamentals of Statistical and Thermal Physics*
- Pathria & Beale, *Statistical Mechanics*
- Chandler, *Introduction to Modern Statistical Mechanics*

---

## Contact

**Hyunje Jun**  
GitHub: [@eigenname](https://github.com/eigenname)  
Email: [available on GitHub profile]

**Issues & Questions:** [GitHub Issues](https://github.com/eigenname/1D-Random-Walk-Simulation/issues)
