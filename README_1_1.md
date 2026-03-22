# BoundWalk — Animated 1D Bounded Random Walk (v1.1)

An extension of [Animation 1.0](https://github.com/your-username/your-1.0-repo-link) — the core architecture, observables, and animation structure are identical. The key change is that step size is no longer fixed but randomly sampled at each step from a uniform distribution $|\Delta X| \sim \mathcal{U}[\Delta x_{min}, \Delta x_{max}]$, making this a **continuous-state** random walk rather than a discrete-grid one.

---

## What Changed From v1.0

| | Animation 1.0 | Animation 1.1 |
|---|---|---|
| Step size | Fixed: `step_size=0.1` | Random: `step_bounds=(0.1, 1.0)` |
| Position space | Discrete grid ($W=11$ states) | Effectively continuous |
| Histogram bins | Natural (bin = reachable state) | Measurement choice (bin width = `step_bounds[0]`) |
| Entropy convergence | Slow — particle diffuses locally | Fast — large steps explore domain immediately |
| KLD convergence | Gradual, monotonic | Steep initial drop, noisier near equilibrium |
| Position vs N plot | Staircase-like | Irregular, time-series-like |

The distinction between **microstates** and **histogram bins** that coincided in v1.0 now separates — bins are purely a measurement resolution, not a count of physically distinct reachable states.

---

## Usage

```python
from boundwalk import BoundWalk, BW

walk = BW(
    total_steps=300,
    step_bounds=(0.1, 1.0),  # uniform sampling range for |ΔX|
    init_pos=0,
    boundaries=[0, 1],
    seed=None
)

walk.visualize()
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `total_steps` | `int` | required | Number of steps $N$ |
| `step_bounds` | `tuple` | `(0.1, 1.0)` | Range for uniform step size sampling $[\Delta x_{min}, \Delta x_{max}]$ |
| `init_pos` | `float` | `0` | Initial position $X_0$ |
| `boundaries` | `list` | `[0, 1]` | Reflecting domain $[a, b]$ |
| `seed` | `int` | `None` | RNG seed for reproducibility |

---

## Notes

- The notebook is committed with outputs cleared. Re-run all cells to regenerate the animation locally.
- GitHub does not render jshtml animations in notebook previews. Use [nbviewer](https://nbviewer.org) by pasting the notebook's GitHub URL to view the full animated output.
- **Next — Animation 1.2**: Gaussian-distributed step sizes, connecting step magnitude variance to temperature and yielding a chi-squared energy distribution as a physically motivated observable.
