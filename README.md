# BoundWalk — 1D Bounded Random Walk Simulator

A Python simulation of a particle performing a **1D random walk confined to a reflecting domain $[a, b]$** with Neumann (fold-back) boundary conditions. At each step, an empirical probability distribution is built from observed positions, and key thermodynamic observables are tracked in real time.

Originally developed for **PHSX 671: Thermal Physics** and extended independently as a project at the intersection of statistical mechanics, information theory, and data science.

---

## Features

- **Reflecting boundaries** — fold-back rule handles arbitrarily large overshoots via a `while` loop, not a single sign flip
- **Random step sizes** — drawn from $|\Delta X| \sim \mathcal{U}(\Delta x_{\min}, \Delta x_{\max})$ each step; step bounds serve as a proxy for temperature
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
| `step_bounds` variance | Proxy for temperature — wider bounds correspond to higher effective $k_B T$ |

The walk is a discrete-time approximation to the **overdamped Langevin equation** — a fully thermalized particle with no inertia, driven purely by noise within a confining potential well.

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version |
|---|---|
| `numpy` | ≥ 1.25.0 |
| `pandas` | ≥ 2.1.0 |
| `matplotlib` | ≥ 3.8.0 |
| `scipy` | ≥ 1.11.0 |
| `ipython` | ≥ 8.15.0 |

---

## Usage

`boundwalk.py` contains only the class definition. Instantiate and visualize from the companion notebook `BW_Animated_1_1.ipynb`:

```python
from boundwalk import BoundWalk

walk = BoundWalk(
    total_steps=200,
    step_bounds=(0.1, 1.0),
    init_pos=0,
    boundaries=[0, 1],
    seed=42
)

walk.visualize()
```

The animation renders inline as an interactive jshtml widget.

---

## Class Reference — `BoundWalk`

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `total_steps` | `int` | required | Number of steps $N$ |
| `step_bounds` | `tuple` | `(0.1, 1.0)` | Range for uniform step size sampling $[\Delta x_{\min}, \Delta x_{\max}]$ |
| `init_pos` | `float` | `0` | Initial position $X_0$ |
| `boundaries` | `list` | `[0, 1]` | Reflecting domain $[a, b]$ |
| `seed` | `int` | `None` | RNG seed for reproducibility |

### Methods

| Method | Description |
|---|---|
| `simulate()` | Generates the walk and computes all per-step observables. Called automatically on instantiation. |
| `visualize()` | Renders the animated jshtml visualization. Must be called explicitly. |
| `_reflect(pos, a, b)` | Static. Fold-back reflection — folds repeatedly until position is inside $[a, b]$. |

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

## Visualization Layout

The animation renders four panels updated each frame:

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

**Histogram details:**
- Bin width fixed to `step_bounds[0]` (the minimum step size)
- Bins centered on the domain grid $[a, b]$ partitioned by bin width
- Y-axis scales dynamically per frame to the current maximum empirical probability
- Dashed reference line marks the uniform PDF $\mathcal{U}(a, b) = 1/W$

**Entropy plot:** Y-axis fixed at $[0, \ln W + 0.1]$; dashed red line marks the Boltzmann supremum $\ln W$

**KLD plot:** Y-axis autoscales from data

**Shared x-axis** (Position / Entropy / KLD plots): starts at $[0, 5]$ and expands as $n$ grows

---

## Changelog

### Animation 1.1 (current)

| | v1.0 | v1.1 |
|---|---|---|
| Step size | Fixed scalar `step_size` | Random: `step_bounds=(min, max)` |
| Position space | Discrete grid ($W = 11$ states) | Effectively continuous |
| Histogram bins | One bin per reachable state | Bin width = `step_bounds[0]` |
| Entropy convergence | Slow — local diffusion | Fast — large steps explore domain immediately |
| KLD convergence | Gradual, monotonic | Steep initial drop, noisier near equilibrium |
| Position vs N | Staircase-like | Irregular time-series |

---

## Roadmap

- **Animation 1.2** — Unified step size interface: `step_dist` parameter accepting a `float`, `tuple`, or `callable` for fixed, uniform, or arbitrary step distributions
- **Animation 1.3** — Velocity tracking via lag-1 displacements; phase space animation $(x_n, v_n)$; speed/energy distribution histogram
- **Animation 1.4** — Gaussian step sizes $\Delta x_n \sim \mathcal{N}(0, \sigma^2)$; connect $\sigma^2$ to temperature; Maxwell-Boltzmann speed distribution
- **Animation 2.0** — Full Langevin dynamics with explicit $\Delta t$, $m$, damping $\gamma$; velocity autocorrelation; MSD crossover from ballistic to diffusive; kinetic temperature as internal consistency check

---

## Repository Structure

```
1D-Random-Walk-Simulation/
├── boundwalk.py               # BoundWalk class definition
├── BW_Animated_1_1.ipynb      # Notebook: instantiation and visualization
├── requirements.txt           # Python dependencies
├── README.md
└── .gitignore                 # Excludes __pycache__/ and *.pyc
```

---

## Environment

Developed on Python 3.13, Anaconda, VS Code with Jupyter extension (Windows).
