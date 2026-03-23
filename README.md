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

## Class Methods

| Method | Type | Description |
|---|---|---|
| `simulate()` | instance | Generates walk and computes per-step observables. Called automatically on instantiation. |
| `visualize()` | instance | Renders the animated visualization inline as jshtml. Call explicitly after instantiation. |
| `_reflect(pos, a, b)` | static | Fold-back reflection rule — handles arbitrarily large boundary overshoots. |
| `_readable_tick_step(bin_width, domain_width)` | static | Derives a clean x-tick spacing from bin width so histogram labels stay readable at any resolution. |
| `_readable_yticks(p_max)` | static | Generates clean y-tick values up to the current maximum probability — scales dynamically per frame. |

---

## Visualization Notes

The histogram and plots adapt intelligently across all panels:

- **Histogram x-ticks** — spacing derived automatically via `_readable_tick_step` so at most 10 labels appear regardless of bin count. For example, `step_bounds[0]=0.01` gives 100 bins but shows clean `0.10` spaced labels rather than 100 overlapping tick marks.
- **Histogram y-axis** — scales dynamically per frame via `_readable_yticks`, tracking the current maximum empirical probability. Early frames show a zoomed-in y-axis; as the distribution flattens toward uniform it rescales accordingly.
- **Histogram x-axis** — fixed at the full domain $[a, b]$ every frame so the complete picture is always visible.
- **Shared x-axis (position, entropy, KLD plots)** — starts at $[0, 5]$ and expands dynamically as $n$ grows beyond 5, driven by the position vs $n$ plot and propagated to the entropy and KLD plots via `sharex`. Entropy y-axis is fixed at $[0, \ln W + 0.1]$; KLD y-axis autoscales from data.
- **KLD and entropy** — initialized with positive-only y-axes, eliminating the garbage axis values that appeared at frame 0 before any data was present.

---

## Notes

- The notebook is committed with outputs cleared. Re-run all cells to regenerate the animation locally.
- GitHub does not render jshtml animations in notebook previews. Use [nbviewer](https://nbviewer.org) by pasting the notebook's GitHub URL to view the full animated output.
- **Next — Animation 1.2**: Unified class supporting both fixed and uniformly randomly sampled step sizes via a single parameter interface, before moving toward Gaussian steps and physical observables like velocity and energy distributions.