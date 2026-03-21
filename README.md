# BoundWalk — Animated 1D Bounded Random Walk

A 1D discrete random walk simulator with reflecting (Neumann) boundary conditions, built in Python. Developed as a personal project at the intersection of statistical mechanics and statistical modeling — tracking the empirical position distribution, Shannon entropy, and KL divergence as the walk evolves toward thermodynamic equilibrium.

---

## Background

This project originated from a statistical mechanics course assignment on bounded random processes. The simulator models a particle performing a random walk confined to an interval $[a, b]$, where boundary collisions are handled by fold-back reflection rather than absorption. The key observables — entropy and KL divergence — connect directly to information-theoretic interpretations of thermalization: as the walk runs, the empirical distribution converges toward the uniform (maximum entropy) distribution, and the system "forgets" its initial condition.

The simulation is intentionally memoryless (each step is i.i.d.), placing it squarely in the **overdamped Langevin limit** — the statistical description of a fully thermalized particle. Future versions will introduce directional memory (persistent random walk) and explicit dynamics (Langevin equation) as intermediate steps toward a physically realistic simulation.

---

## Features

- **Reflecting boundary conditions** via a robust fold-back rule that handles arbitrarily large step overshoots
- **Empirical probability histogram** tracking the evolving position distribution at each step
- **Shannon entropy** $H[X_n] = -\sum_i p_i \ln p_i$ plotted against $n$, with the Boltzmann supremum $\ln W$ as a reference
- **KL divergence** $D_{KL}(P \| Q)$ measured against the uniform reference distribution, converging to zero at equilibrium
- **Animated visualization** of all observables simultaneously via `matplotlib.FuncAnimation`, rendered inline as jshtml in Jupyter
- **Decoupled simulation and visualization** — `simulate()` and `visualize()` are separate methods, allowing data inspection without triggering a render
- **Reproducible runs** via optional `seed` parameter

---

## File Structure

```
├── boundwalk.py          # BoundWalk class definition
├── BW_Animated_1_0.ipynb # Notebook: object instantiation and animation output
└── README.md
```

---

## Usage

```python
from boundwalk import BoundWalk, BW

# instantiate
walk = BW(
    total_steps=300,
    step_size=0.1,      # fixed step size
    init_pos=0,         # initial position
    boundaries=[0, 1],  # reflecting domain
    seed=None           # set for reproducibility
)

# inspect data
display(walk.data)

# render animation
walk.visualize()
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `total_steps` | `int` | required | Number of steps $N$ |
| `step_size` | `float` | `0.1` | Fixed displacement magnitude $\|\Delta X\|$ |
| `init_pos` | `float` | `0` | Initial position $X_0$ |
| `boundaries` | `list` | `[0, 1]` | Reflecting domain $[a, b]$ |
| `seed` | `int` | `None` | RNG seed for reproducibility |

---

## Simulation Data

Each instantiation generates a `walk.data` DataFrame with one row per step:

| Column | Description |
|---|---|
| `n ≤ N` | Step index |
| `nth Displacement` | Realized displacement after reflection |
| `nth Position` | Position $X_n$ |
| `nth Possible Outcomes` | Distinct positions visited up to step $n$ |
| `nth Probabilities` | Empirical probabilities of each visited position |
| `nth Entropy` | Shannon entropy $H[X_n]$ at step $n$ |
| `nth KLD` | $D_{KL}$ of empirical distribution vs uniform |

---

## Observable Behavior

With default parameters (`step_size=0.1`, `boundaries=[0, 1]`), the domain has $W = 11$ equally spaced microstates. The simulation demonstrates:

- **Entropy** climbing from $H = 0$ (certain initial state) toward the Boltzmann supremum $\ln(11) \approx 2.3979$
- **KLD** decaying from its maximum (concentrated distribution at $X_0$) toward zero (uniform distribution)
- **Empirical histogram** converging toward the uniform reference $U(0, 1)$ as $n \to N$

The pace of convergence depends on $N$ — larger $N$ gives cleaner convergence at the cost of a larger animation embed.

---

## Dependencies

```
numpy
pandas
matplotlib
scipy
IPython
```

---

## Notes

- The notebook (`BW_Animated_1_0.ipynb`) is committed with outputs cleared to keep file size manageable. Re-run all cells to regenerate the animation locally.
- GitHub does not render jshtml animations in notebook previews. Use [nbviewer](https://nbviewer.org) to view the full animated output by pasting the notebook's GitHub URL.
- This is **Animation 1.0** — fixed step size, memoryless walk. A subsequent version (`Animation 2.0`) will introduce random step sizes drawn from a uniform distribution, moving toward a continuous-state approximation with richer distributional behavior.
