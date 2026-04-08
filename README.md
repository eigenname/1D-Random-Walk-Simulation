# BoundWalk — 1D Bounded Random Walk Simulator

A Python simulation of a particle performing a **1D random walk confined to a reflecting domain $[a, b]$** with Neumann (fold-back) boundary conditions. At each step, an empirical probability distribution is built from observed positions, and key thermodynamic observables are tracked in real time.

Originally developed for **PHSX 671: Thermal Physics** and extended independently as a project at the intersection of statistical mechanics, information theory, and data science.

---

## Features

- **Unified step size interface** — fixed scalar, uniform random $\mathcal{U}[a, b]$, or standard normal $\mathcal{N}(\mu, \sigma)$ step sizes via the `Uniform()` and `Normal()` factory functions in `bw_tools.py`
- **Reflecting boundaries** — fold-back rule handles arbitrarily large overshoots via a `while` loop, not a single sign flip
- **Per-step observables** — Shannon entropy $H[X_n]$, KL divergence $D_{KL}(\mathbb{P} \| \mathcal{U})$, and empirical position histogram, all updated frame by frame
- **Animated inline visualization** — interactive jshtml widget rendered directly in Jupyter, no external player needed
- **Reproducible runs** — optional `seed` parameter for the NumPy default RNG

---

## Physics Background

| Observable | Description |
|---|---|
| $H[X_n] = -\sum_i p_i \ln p_i$ | Shannon entropy — climbs toward the Boltzmann supremum $\ln W$ as the system thermalizes |
| $D_{KL}(\mathbb{P} \| \mathcal{U})$ | KL divergence from the uniform (maximum entropy) reference — converges to zero at equilibrium |
| $\ln W$ | Boltzmann entropy supremum — maximum entropy for $W$ equally accessible microstates |
| Step size distribution | Proxy for temperature — wider or higher-variance distributions correspond to higher effective $k_B T$ |

The walk is a discrete-time approximation to the **overdamped Langevin equation** — a fully thermalized particle with no inertia, driven purely by noise within a confining potential.

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version |
|---|---|
| `numpy` | ≥ 1.26.0 |
| `pandas` | ≥ 2.1.0 |
| `matplotlib` | ≥ 3.8.0 |
| `scipy` | ≥ 1.11.0 |
| `ipython` | ≥ 8.15.0 |

---

## Usage

`boundwalk.py` contains the `BoundWalk` class. `bw_tools.py` contains all helper functions and step size factory functions. Instantiate from the companion notebook `BW_Animated_1_2.ipynb`:

```python
from boundwalk import BoundWalk as BW
from bw_tools import Uniform, Normal

# Fixed step size
walk_fixed = BW(total_steps=100, step_size=0.1)

# Uniform random step size
walk_uniform = BW(total_steps=100, step_size=Uniform(0.1, 1.0, bin_width=0.1))

# Standard normal step size (when implemented)
# walk_normal = BW(total_steps=100, step_size=Normal(0, 0.3, bin_width=0.05))

walk_fixed.__visualize__()
```

The animation renders inline as an interactive jshtml widget.

---

## Class Reference — `BoundWalk`

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `total_steps` | `int` | required | Number of steps $N$ |
| `step_size` | `float`, `int`, or `dict` | required | Step size — fixed scalar or dict from `Uniform()`/`Normal()` factory |
| `initial_position` | `float` | `0` | Initial position $X_0$ |
| `boundaries` | `tuple` | `(0, 1)` | Reflecting domain $[a, b]$ |
| `seed` | `int` | `None` | RNG seed for reproducibility |
| `fps` | `int` | `500` | Millisecond interval between animation frames |

### Methods

| Method | Description |
|---|---|
| `__simulate__()` | Generates the walk and computes all per-step observables. Called automatically on instantiation. |
| `__visualize__()` | Renders the animated jshtml visualization. |

### `self.data` — DataFrame Schema

| Column | Description |
|---|---|
| `n ≤ N` | Step index |
| `nth Displacement` | Realized displacement after reflection |
| `nth Position` | Position $X_n$ |
| `nth Possible Outcomes` | Distinct positions visited up to step $n$ |
| `nth Probabilities` | Empirical probabilities of each visited position |
| `nth Entropy` | Shannon entropy $H[X_n]$ at step $n$ |
| `nth KLD` | $D_{KL}$ of empirical distribution vs uniform at step $n$ |

---

## `bw_tools.py` — Helper Functions

| Function | Description |
|---|---|
| `Uniform(a, b, bin_width)` | Factory — returns a uniform step size dict $\mathcal{U}[a, b]$ with given histogram bin width |
| `Normal(mean, std_dev, bin_width)` | Factory — returns a normal step size dict $\mathcal{N}(\mu, \sigma)$ with given histogram bin width |
| `find_rounding_precision(step_size)` | Derives rounding precision from step size parameters |
| `create_displacements(seed, total_steps, step_size)` | Generates displacement array based on step size type |
| `get_bin_width(step_size)` | Extracts bin width from step size parameters |
| `define_domain(left_bound, right_bound, step_size)` | Builds the domain grid $[a, b]$ partitioned by bin width |
| `reflect(pos, left_bound, right_bound)` | Fold-back reflection — folds repeatedly until position is inside $[a, b]$ |
| `pad_to_domain(outcomes, probs, domain, left_bound, delta)` | Aligns empirical distribution onto domain grid for KLD computation |

---

## Visualization Layout

```
┌─────────────────────┬─────────────────────┐
│  1D Position (live) │  P(Xₙ) Histogram    │
├─────────────────────┴─────────────────────┤
│           Position Xₙ vs n → N            │
├────────────────────────────────────────────┤
│           Entropy H[Xₙ] vs n → N          │
├────────────────────────────────────────────┤
│         KL Divergence D_KL vs n → N       │
└────────────────────────────────────────────┘
```

**Histogram:** bin width set by `bin_width` parameter in `Uniform()`/`Normal()`, or by `step_size` for fixed walks. Y-axis scales dynamically per frame. Dashed red line marks $\mathcal{U}(a, b) = 1/W$.

**Entropy plot:** Y-axis fixed at $[0, \ln W + 0.1]$; dashed red line marks the Boltzmann supremum $\ln W$.

**KLD plot:** Y-axis autoscales from data.

**Shared x-axis** (Position / Entropy / KLD): starts at $[0, 5]$ and expands as $n$ grows.

---

## Changelog

### Animation 1.2 (current)

- Unified `step_size` interface replacing `step_bounds` — accepts fixed scalar, `Uniform()`, or `Normal()` dict
- `bw_tools.py` extracted as a separate module containing all helper and factory functions
- `@dataclass(kw_only=True)` replacing manual `__init__`; `__post_init__` handles simulation setup
- `self.domain` and `self.bin_width` stored on instance during `__simulate__`, shared with `__visualize__`
- `bin_width` parameter added to `Uniform()` and `Normal()` factories for explicit histogram control
- `fps` parameter added for animation speed control

### Animation 1.1

- Random step sizes via `step_bounds=(min, max)` replacing fixed `step_size`
- Histogram bin width tied to `step_bounds[0]`; dynamic y-axis scaling per frame
- Fold-back reflection `while` loop replacing single sign flip
- Decoupled `simulate()` / `visualize()`

---

## Roadmap

- **Animation 1.3** — Velocity tracking via lag-1 displacements; phase space animation $(x_n, v_n)$; speed/energy distribution histogram
- **Animation 1.4** — Gaussian step sizes $\Delta x_n \sim \mathcal{N}(0, \sigma^2)$; connect $\sigma^2$ to temperature; Maxwell-Boltzmann speed distribution
- **Animation 2.0** — Full Langevin dynamics with explicit $\Delta t$, $m$, damping $\gamma$; velocity autocorrelation; MSD crossover from ballistic to diffusive; kinetic temperature as internal consistency check

---

## Repository Structure

```
1D-Random-Walk-Simulation/
├── boundwalk.py               # BoundWalk class definition
├── bw_tools.py                # Helper and factory functions
├── BW_Animated_1_2.ipynb      # Notebook: instantiation and visualization
├── requirements.txt           # Python dependencies
├── README.md
└── .gitignore                 # Excludes __pycache__/ and *.pyc
```

---

## Environment

Developed on Python 3.13, Anaconda, VS Code with Jupyter extension (Windows).
