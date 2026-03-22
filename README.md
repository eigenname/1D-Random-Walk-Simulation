# Random Walk Simulator — Static Visualization

A 1D discrete random walk simulator built in Python, developed as a class project for PHSX 671: Thermal Physics and extended independently. Simulates unit step size walks with equal probability of stepping left or right, starting at the origin by default. The project is organized into two stages, each building conceptually on the previous to connect probabilistic dynamics, statistical scaling laws, and information-theoretic observables.

---

## Terminology

Three levels of simulation structure are used throughout. Each level aggregates the one below it:

**Walk** — a single random walk of $N$ steps. The fundamental unit. Produces one trajectory, one position distribution.

**Trial** — a collection of multiple walks, all sharing the same step count $N$ and all starting at the origin. Think of it as a physics experiment: run the same experiment $M$ times under identical conditions and record each outcome. The trial's observable is the **net displacement** of each walk — where did each walker end up after $N$ steps? From this, the expected net displacement $\langle X_N \rangle$ and its standard deviation $\sigma_N$ are estimated empirically across the $M$ walks.

**List** — a collection of trials at increasing step counts $N_1 < N_2 < \cdots$. Like repeating the same experiment at different parameter settings. Enables studying how trial-level statistics (expected displacement, standard deviation, entropy) scale as $N$ grows large.

---

## Stage 1 — Walk and Trial Simulations

### Single Walk
Simulates one random walk of $N$ steps. Visualizations include:
- Position vs step count plot
- Histogram of positions visited
- Box and whisker plot of positions

### Trial (Ensemble of Walks)
Simulates $M$ walks of fixed step count $N$. Key results demonstrated empirically:
- Expected net displacement $\langle X_N \rangle \approx 0$ — the walk is unbiased
- Standard deviation $\sigma_N \approx \sqrt{N}$ — displacement spreads as the square root of step count
- Net displacement distribution approximated by a Gaussian model, improving in accuracy as $N$ grows — a direct demonstration of the Central Limit Theorem

Visualizations include:
- Histogram and box plot of net displacements across all walks in the trial
- Position vs step count for multiple walks simultaneously
- Net displacement vs number of walks, with $\pm\sqrt{N}$ convergence band showing the standard deviation settling toward $\sqrt{N}$

### List (Sweep Over Step Counts)
Runs a series of trials at increasing $N$. Demonstrates that variance scales linearly with step count by plotting $\sigma_N$ vs $N$ — the standard deviations follow a positive $\sqrt{N}$ trend, confirming that the variance of a random walk is directly proportional to step count.

---

## Stage 2 — Entropy and Bounded Walks

### Entropy
Shannon entropy $H = -\sum_i p_i \ln p_i$ is introduced as an observable at both levels:
- **Entropy vs position** for a single walk — how uncertainty over the walker's position evolves with step count
- **Entropy vs net displacement** for a trial — how uncertainty over outcomes is distributed across the ensemble

### Bounded Walks
Symmetric reflecting boundaries are imposed around the origin, confining positions and net displacements to a finite interval $[-L, L]$. The same walk, trial, and list visualizations from Stage 1 are reproduced under this constraint.

### Entropy Scaling — The Culminating Result
Each trial in a list has a maximum entropy $S_{max}$ associated with its net displacement distribution. Plotting $S_{max}$ against step count $N$ reveals that **maximal entropy grows logarithmically** with step count:

$$S_{max} \sim \ln N$$

This is fitted using a logarithmic series approximation up to the 16th-order term. The result connects directly to Boltzmann's entropy formula $S = k_B \ln W$, where $W$ is the number of accessible microstates — here growing with $N$ as the bounded walk explores more of its domain.

As a consequence of the boundaries, the standard deviation also becomes logarithmic rather than following the $\sqrt{N}$ scaling of the unbounded case, and is fitted with the same log series model.

---

## Relationship to Other Versions

This is the **original static visualization** branch — no animation, no KL divergence tracking, no real-time observables. It represents the foundational project from which the animated versions were developed:

| Version | Description |
|---|---|
| **This branch** | Static plots, full walk/trial/list hierarchy, entropy scaling law |
| **Animated 1.0** | Real-time animation, fixed step size, Shannon entropy + KLD tracked per step |
| **Animated 1.1** | Same as 1.0 but with uniformly randomly sampled step size |

The logarithmic entropy result demonstrated here — that $S_{max} \sim \ln N$ for bounded walks — is the same relationship the animated versions demonstrate in real time through the entropy plot converging toward the Boltzmann supremum $\ln W$.

---

## Dependencies

```
numpy
pandas
matplotlib
scipy
```

---

## Notes

- All walks start at the origin by default unless otherwise specified.
- The Gaussian approximation to the net displacement distribution improves as $N$ increases — this is the Central Limit Theorem in action, and is most visible in the trial-level histogram.
- The 16th-order logarithmic series fit to $S_{max}$ vs $N$ is chosen empirically as the best fitting model at the step counts simulated; lower-order fits are also available and may be sufficient for smaller $N$.
