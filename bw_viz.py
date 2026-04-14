"""
BoundWalk Visualization Module
Provides visualization functions for different observables from BoundWalk simulations
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, expon
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
from IPython.display import display, HTML


def visualize_position(walk, verbose=None):
    """
    Animate position observable: X(t) trajectory + histogram + entropy + KLD
    
    Parameters
    ----------
    walk : BoundWalk instance
        Completed simulation with .data DataFrame
    verbose : bool, optional
        Override walk.verbose setting
    """
    if verbose is None:
        verbose = walk.verbose
    
    if verbose:
        print("[BoundWalk] Building position animation...")
    
    # Extract data
    N = walk.total_steps
    init_position = walk.initial_position
    left_bound, right_bound = walk.boundaries
    bin_width = walk.bin_width
    centers = walk.domain
    edges = np.append(centers - bin_width/2, centers[-1] + bin_width/2)
    
    # Pre-extract numpy arrays (faster than DataFrame indexing in loop)
    positions_np = walk.data['nth Position'].to_numpy()
    times_np = walk.data['t'].to_numpy()
    entropies_np = walk.data['nth Entropy'].to_numpy()
    klds_np = walk.data['nth KLD'].to_numpy()
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 1], hspace=0.6)
    
    # ======================== Subplot 1: Particle Animation ========================
    particle_animation = fig.add_subplot(gs[0, 0])

    position_marker, = particle_animation.plot([], [], 'o', markersize=6, color='C0', label=r"$\vec{x}_{_{t}}$")
    step_size = walk.step_size
    if isinstance(step_size, float): # display step size info in title, whether fixed 
        step_info = f"|Δ\vec{{X}}| = {step_size}"
    else: # or randomly sampled from distributions
        if step_size['type'] == 'uniform':
            step_info = rf"|Δ\vec{{X}}| \sim \mathcal{{U}}[{step_size['bounds'][0]}, {step_size['bounds'][1]}]"
        else:  # normal
            step_info = rf"|Δ\vec{{X}}| \sim \mathcal{{N}}({step_size['params'][0]}, {step_size['params'][1]**2})"
    
    particle_animation.set_title(rf"$\mathcal{{BW}}: N={N}, {step_info}, \vec{{X}}_{{_{{0}}}}={init_position}$")
    particle_animation.get_yaxis().set_visible(False)
    particle_animation.set_xlabel(r"$\vec{X}_{_{t}} = \vec{x}_{_{t}}$")
    particle_animation.set_ylim(-0.05, 0.1)
    particle_animation.set_xlim(left_bound, right_bound)
    particle_animation.legend(loc='upper left')
    
    # ======================== Subplot 2: Histogram ========================
    position_histogram = fig.add_subplot(gs[0, 1])
    bars = position_histogram.bar(centers, np.zeros_like(centers), width=bin_width, align='center', color="C0", edgecolor="black")
    position_histogram.axhline(1/ len(centers), color="black", linestyle="--", linewidth=1, label=rf"$U[{left_bound},{right_bound}]$")
    
    position_histogram.set_title(r"$\mathbb{P}(\vec{X}_{_{t}})$ vs $\vec{X}_{_{t}}$")
    position_histogram.set_ylabel(r"$\mathbb{P}(\vec{X}_{_{t}})$")
    position_histogram.set_xlabel(r"$\vec{X}_{_{t}} = \vec{x}_{_{t}}$")
    position_histogram.set_xlim(left_bound, right_bound)
    position_histogram.legend(loc='upper right')
    
    # ======================== Subplot 3: Position vs Time ========================
    position_subplot = fig.add_subplot(gs[1, :])
    position_trajectory, = position_subplot.plot([], [], color="C0")
    trajectory_marker, = position_subplot.plot([], [], ".", color="C0", label=r"$\vec{x}_{_{t}}$")
    
    position_subplot.set_title(r"$\vec{X}_{_{t}}$ vs $t$")
    position_subplot.set_ylabel(r"$\vec{X}_{_{t}} = \vec{x}_{_{t}}$")
    position_subplot.set_ylim(left_bound, right_bound)
    position_subplot.set_xlim(0, 10)
    position_subplot.tick_params(labelbottom=False)
    position_subplot.legend(loc='upper left')
    
    # ======================== Subplot 4: Entropy ========================
    entropy_subplot = fig.add_subplot(gs[2, :], sharex=position_subplot)
    entropy_trajectory, = entropy_subplot.plot([], [], color="C0")
    entropy_marker, = entropy_subplot.plot([], [], ".", color="C0", label=r"$S_{_{G}}[\vec{{x}}_t]$")
    entropy_subplot.axhline(np.log(len(centers)), color='black', linewidth=1, linestyle='--', label=rf"$S_{{_{{B}}}} = \ln({len(centers)})$")
    
    entropy_subplot.set_title(r"$S_{_{G}}[\vec{{X}}_t]$ vs $t$")
    entropy_subplot.set_ylabel(r"$S_{_{G}}[\vec{X}_t]$")
    entropy_subplot.set_ylim(0, np.log(len(centers)) + 0.5)
    entropy_subplot.tick_params(labelbottom=False)
    entropy_subplot.legend(loc='upper left')
    
    # ======================== Subplot 5: KL Divergence ========================
    kld_subplot = fig.add_subplot(gs[3, :], sharex=position_subplot)
    kld_trajectory, = kld_subplot.plot([], [], color="C0")
    kld_marker, = kld_subplot.plot([], [], ".", color="C0", label=r"$D_{{_{{KL}}}}$")
    
    kld_subplot.set_title(r"$D_{{_{{KL}}}}(\mathbb{P}||U)$ vs $t$")
    kld_subplot.set_ylabel(r"$D_{{_{{KL}}}}(\mathbb{P}||U)$")
    kld_subplot.set_xlabel(r"$t \to \infty$")
    kld_subplot.set_ylim(0, np.log(len(centers)) + 0.5)
    kld_subplot.legend(loc='upper left')
    
    # ======================== Animation Function ========================
    milestones = {int(len(walk.data) * p) for p in [0.25, 0.5, 0.75, 1.0]} if verbose else set()
    def _animate(frame):
        if frame in milestones:
            print(f"[BoundWalk] Rendering: {frame}/{len(walk.data)} ({100*frame//len(walk.data)}%)")
        
        # Particle position
        position_marker.set_data([positions_np[frame]], [0])

        # Position histogram
        if frame == 0: # delta distribution at initial position
            probs = np.zeros(len(bars)) # initialize 0 vector of len of histogram values
            probs[np.argmin(np.abs(centers - init_position))] = 1.0 # determine index of initial_position and 

        else: # compute histogram probabilities based on positions up to current frame
            current_positions = positions_np[1:frame+1]
            counts, _ = np.histogram(current_positions, bins=edges)
            probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)

        for bar, height in zip(bars, probs): # update histogram bars
            bar.set_height(height)

        position_histogram.set_ylim(0, max(probs.max(), 1/len(centers) * 1.5) * 1.05) # never let ylim drop below ~1.5x the uniform line
        yticks = np.linspace(0, max(probs.max(), 1/len(centers) * 1.5), 5)
        position_histogram.set_yticks(yticks)
        position_histogram.set_yticklabels([f"{y:.3f}" for y in yticks])

        # Position trajectory
        position_trajectory.set_data(times_np[:frame+1], positions_np[:frame+1])
        trajectory_marker.set_data([times_np[frame]], [positions_np[frame]])
        
        x_max = times_np[frame] if frame > 2 else 2
        position_subplot.set_xlim(0, x_max)
        position_subplot.xaxis.set_major_locator(MaxNLocator(integer=False, prune='both', nbins=6))
        
        # Entropy
        if frame > 0:
            entropy_trajectory.set_data(times_np[1:frame+1], entropies_np[1:frame+1])
            entropy_marker.set_data([times_np[frame]], [entropies_np[frame]])
            # ax_entr.set_ylim(entropies_np[:frame+1].min() - 0.1*entropies_np[:frame+1].max(), entropies_np[:frame+1].max() + 0.1*entropies_np[:frame+1].max())
        
        # KLD
        if frame > 0:
            kld_trajectory.set_data(times_np[1:frame+1], klds_np[1:frame+1])
            kld_marker.set_data([times_np[frame]], [klds_np[frame]])
    
    plt.subplots_adjust(left=0.075, bottom=0.075, hspace=0.4)
    plt.close(fig)
    
    animation = FuncAnimation(fig, _animate, frames=len(walk.data), 
                        interval=walk.ms_between_frames, blit=False)
    
    if verbose:
        print("[BoundWalk] Rendering animation...")
    matplotlib.rcParams["animation.embed_limit"] = 50_000_000 # adjust (RC) for increasing animation file size to ~50 MB, may need to turn off for gif creation!
    display(HTML(animation.to_jshtml()))
    if verbose:
        print("[BoundWalk] Position animation complete.")


def visualize_momentum(walk, verbose=None):
    """
    Animate momentum observable: P(t) trajectory + distribution + autocorrelation
    
    Parameters
    ----------
    walk : BoundWalk instance
        Completed simulation with .data DataFrame
    verbose : bool, optional
        Override walk.verbose setting
    """
    if verbose is None:
        verbose = walk.verbose
    
    if verbose:
        print("[BoundWalk] Building momentum animation...")
    
    # Extract data
    positions_np = walk.data['nth Position'].to_numpy()  
    momenta_np = walk.data['nth Momentum'].to_numpy()
    times_np = walk.data['t'].to_numpy()
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.6)
    
    # ======================== Subplot 1: Phase Space (X, P) ========================
    phase_space = fig.add_subplot(gs[0])

    x_min, x_max = positions_np.min(), positions_np.max()
    p_min, p_max = momenta_np.min(), momenta_np.max()
    p_grid = np.linspace(p_min + 0.1*p_min, p_max + 0.1*p_max, 1000)
    phase_point, = phase_space.plot([], [], ".", color="C4")
    phase_traj, = phase_space.plot([], [], color="C4", alpha=0.25)
    
    phase_space.hlines(y=p_min, xmin=x_min, xmax=x_max, color='C3', linewidth=0.7, linestyle='--')
    phase_space.hlines(y=p_max, xmin=x_min, xmax=x_max, color='C3', linewidth=0.7, linestyle='--')
    phase_space.vlines(x=x_min, ymin=p_min, ymax=p_max, color='blue', linewidth=0.7, linestyle='--')
    phase_space.vlines(x=x_max, ymin=p_min, ymax=p_max, color='blue', linewidth=0.7, linestyle='--')

    phase_space.set_title(r"Phase Space: $(\vec{X}_{_{t}}, \vec{P}_{_{t}})$")
    phase_space.set_xlabel(r"$\vec{X}_{_{t}} = \vec{x}_{_{t}}$")
    phase_space.set_ylabel(r"$\vec{P}_{_{t}} = \vec{p}_{_{t}}$")
    phase_space.set_xlim(x_min - 0.1*x_max, x_max + 0.1*x_max)

    # ======================== Subplot 2: Momentum Trajectory ========================
    momenta_subplot = fig.add_subplot(gs[1])

    momenta_trajectory, = momenta_subplot.plot([], [], color="C3", alpha=0.7)
    momenta_marker, = momenta_subplot.plot([], [], ".", color="C3", label=r"$\vec{P}_{_{t}}$")

    momenta_subplot.set_title(r"$\vec{P}_{_{t}}$ vs $t$")
    momenta_subplot.set_ylabel(r"$\vec{P}_{_{t}} = \vec{p}_{_{t}}$")
    momenta_subplot.set_xlabel(r"$t \to \infty$")
    momenta_subplot.set_xlim(0, 10)
    momenta_subplot.legend(loc='upper left')
    
    # ======================== Subplot 3: Momentum Distribution ========================
    momenta_distribution = fig.add_subplot(gs[2])
    kde_momenta, = momenta_distribution.plot([], [], color="C3", lw=1, label=r'KDE($\vec{P}_{_{t}}$)')
    normal_line, = momenta_distribution.plot([], [], 'k--', linewidth=1, label=rf'$\mathcal{{N}}(\mu_t, \sigma_t^2)$')

    momenta_distribution.set_title(r"$\rho(\vec{P}_{_{t}})$ vs $\vec{P}_{_{t}}$")
    momenta_distribution.set_ylabel(r"$\rho(\vec{P}_{_{t}})$")
    momenta_distribution.set_xlabel(r"$\vec{P}_{_{t}} = \vec{p}_{_{t}}$")
    momenta_distribution.set_xlim(p_grid.min(), p_grid.max())
    momenta_distribution.legend(loc='upper left')
    
    # ======================== Animation Function ========================
    milestones = {int(len(walk.data) * p) for p in [0.25, 0.5, 0.75, 1.0]} if verbose else set()
    
    def _animate(frame):
        if frame in milestones:
            print(f"[BoundWalk] Rendering: {frame}/{len(walk.data)} ({100*frame//len(walk.data)}%)")
        
        # Phase space
        if frame > 0:
            phase_point.set_data(positions_np[:frame+1], momenta_np[:frame+1])
            phase_traj.set_data(positions_np[:frame+1], momenta_np[:frame+1])
            
            phase_space.set_ylim(momenta_np[:frame+1].min() * 1.1,momenta_np[:frame+1].max() * 1.1)

        # Momentum trajectory
        momenta_trajectory.set_data(times_np[:frame+1], momenta_np[:frame+1])
        momenta_marker.set_data([times_np[frame]], [momenta_np[frame]])
        ymin, ymax = momenta_np[:frame+1].min(), momenta_np[:frame+1].max()
        if ymin == ymax:
            pad = 1e-6 if ymin == 0 else abs(ymin) * 0.1
        else:
            pad = 0.1 * (ymax - ymin)

        momenta_subplot.set_ylim(ymin - pad, ymax + pad)
        x_max = times_np[frame] if frame > 2 else 2
        momenta_subplot.set_xlim(0, x_max)

        # Momentum distribution
        for coll in list(momenta_distribution.collections): # Clear any previous bar collections
            coll.remove()
        
        if frame == 0: # Dirac delta at initial momentum (usually 0)
            kde_momenta.set_data([], [])  # Hide KDE
            normal_line.set_data([], [])  # Hide normal
            _ = momenta_distribution.stem([walk.initial_momentum], [1000], linefmt='C3-', markerfmt=' ', basefmt=' ')
            momenta_distribution.set_ylim(0, 1000)
            
        elif frame == 1: # Dirac delta at first observed momentum
            kde_momenta.set_data([], [])  # Hide KDE
            normal_line.set_data([], [])  # Hide normal
            _ = momenta_distribution.stem([momenta_np[1]], [1000], linefmt='C3-', markerfmt=' ', basefmt=' ')
            momenta_distribution.set_ylim(0, 1000)
            
        elif frame > 1: # KDE for frame > 1 
            momenta_subset = momenta_np[1:frame+1]
            p_std_subset = momenta_subset.std()
            p_mean_subset = momenta_subset.mean()
            
            # Update normal distribution based on current data
            normal_density = norm.pdf(p_grid, loc=p_mean_subset, scale=p_std_subset)
            normal_line.set_data(p_grid, normal_density)
            
            # Simple check: if std is reasonable, compute KDE
            if p_std_subset > 1e-10:
                # kde = gaussian_kde(momenta_subset, bw_method=0.1 * p_std_subset * len(momenta_subset)**(-1/5))
                kde = gaussian_kde(momenta_subset, bw_method=0.05) 
                p_density = kde(p_grid)
                kde_momenta.set_data(p_grid, p_density)
                momenta_distribution.fill_between(p_grid, 0, p_density, color='C3', alpha=0.3) # Fill between for visibility
 
                # Update ylim to show both KDE and normal reference
                max_density = max(p_density.max(), normal_density.max())
                momenta_distribution.set_ylim(0, max_density * 1.1)
            else:
                # If no variation, just hide KDE (shouldn't happen often)
                kde_momenta.set_data([], [])
                momenta_distribution.set_ylim(0, normal_density.max() * 1.1)

    plt.subplots_adjust(left=0.1, bottom=0.1, hspace=0.4)
    plt.close(fig)
    
    animation = FuncAnimation(fig, _animate, frames=len(walk.data),
                        interval=walk.ms_between_frames, blit=False)
    
    if verbose:
        print("[BoundWalk] Rendering animation...")
    matplotlib.rcParams["animation.embed_limit"] = 50_000_000 # adjust (RC) for increasing animation file size to ~50 MB, may need to turn off for gif creation!
    display(HTML(animation.to_jshtml()))
    if verbose:
        print("[BoundWalk] Momentum animation complete.")


def visualize_energy(walk, verbose=None):
    """
    Animate energy observable: E(t) conservation + distribution
    
    Parameters
    ----------
    walk : BoundWalk instance
        Completed simulation with .data DataFrame
    verbose : bool, optional
        Override walk.verbose setting
    """
    if verbose is None:
        verbose = walk.verbose
    
    if verbose:
        print("[BoundWalk] Building energy animation...")
    
    # Extract data
    N = walk.total_steps
    energies_np = walk.data['nth Energy'].to_numpy()
    times_np = walk.data['t'].to_numpy()
    positions_np = walk.data['nth Position'].to_numpy()
    momenta_np = walk.data['nth Momentum'].to_numpy()
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.6)
    
    # Subplot 1: Phase Space
    phase_space = fig.add_subplot(gs[0])

    x_min, x_max = positions_np.min(), positions_np.max()
    p_min, p_max = momenta_np.min(), momenta_np.max()
    phase_point, = phase_space.plot([], [], ".", color="C4")
    phase_traj, = phase_space.plot([], [], color="C4", alpha=0.25)
    
    phase_space.hlines(y=p_min, xmin=x_min, xmax=x_max, color='C3', linewidth=0.7, linestyle='--')
    phase_space.hlines(y=p_max, xmin=x_min, xmax=x_max, color='C3', linewidth=0.7, linestyle='--')
    phase_space.vlines(x=x_min, ymin=p_min, ymax=p_max, color='blue', linewidth=0.7, linestyle='--')
    phase_space.vlines(x=x_max, ymin=p_min, ymax=p_max, color='blue', linewidth=0.7, linestyle='--')

    phase_space.set_title(r"Phase Space: $(\vec{X}_{_{t}}, \vec{P}_{_{t}})$")
    phase_space.set_xlabel(r"$\vec{X}_{_{t}} = \vec{x}_{_{t}}$")
    phase_space.set_ylabel(r"$\vec{P}_{_{t}} = \vec{p}_{_{t}}$")
    phase_space.set_xlim(x_min - 0.1*x_max, x_max + 0.1*x_max)

    # Subplot 2: Energy vs Time
    energy_subplot = fig.add_subplot(gs[1])

    energy_trajectory, = energy_subplot.plot([], [], color="C4")
    energy_marker, = energy_subplot.plot([], [], ".", color="C4", label=r"$E_{_{t}}$")
        
    energy_subplot.set_title(r"$E_{_{t}} = \frac{\vec{P}_{t}^2}{2}$ vs $t$")
    energy_subplot.set_ylabel(r"$E_{_{t}}$")
    energy_subplot.set_xlim(0, 10)
    energy_subplot.tick_params(labelbottom=False)
    energy_subplot.legend(loc='upper left')

    # Subplot 3: Energy Distribution 
    energy_distribution = fig.add_subplot(gs[2])
    e_min, e_max = energies_np.min(), energies_np.max()
    e_grid = np.linspace(e_min, e_max, 1000)
    kde_energy, = energy_distribution.plot([], [], color='C4', lw=1, label=r'KDE($E_{_{t}}$)')
    energy_line, = energy_distribution.plot([], [], 'k--', linewidth=1, label=r'Exp$(\lambda)$')

    energy_distribution.set_title(r"$\rho(E_{_{t}})$ vs $E_{_{t}}$")
    energy_distribution.set_ylabel(r"$\rho(E_{_{t}})$")
    energy_distribution.set_xlabel(r"$E_{_{t}} = e_{_{t}}$")
    energy_distribution.set_xlim(e_min, e_max)
    energy_distribution.legend(loc='upper right')

    # ======================== Animation Function ========================
    milestones = {int(len(walk.data) * p) for p in [0.25, 0.5, 0.75, 1.0]} if verbose else set()
    
    def _animate(frame):
        if frame in milestones:
            print(f"[BoundWalk] Rendering: {frame}/{len(walk.data)} ({100*frame//len(walk.data)}%)")
        
        # 1: Phase space
        if frame > 0:
            phase_point.set_data(positions_np[:frame+1], momenta_np[:frame+1])
            phase_traj.set_data(positions_np[:frame+1], momenta_np[:frame+1])
            
            phase_space.set_ylim(momenta_np[:frame+1].min() * 1.1,momenta_np[:frame+1].max() * 1.1)

        # 2: Energy vs time
        energy_trajectory.set_data(times_np[:frame+1], energies_np[:frame+1])
        energy_marker.set_data([times_np[frame]], [energies_np[frame]])
        
        ymin, ymax = energies_np[:frame+1].min(), energies_np[:frame+1].max() 
        if ymin == ymax:
            pad = 1e-6 if ymin == 0 else abs(ymin) * 0.1
        else:
            pad = 0.1 * (ymax - ymin)

        energy_subplot.set_ylim(ymin, ymax + pad)
        x_max = times_np[frame] if frame > 2 else 2 
        energy_subplot.set_xlim(0, x_max)

        # 3: Energy distribution
        for coll in list(energy_distribution.collections): # clear any previous bar collections
            coll.remove()

        if frame == 0: # Dirac delta at initial energy 
            kde_energy.set_data([], []) # Hide KDE
            _ = energy_distribution.stem([walk.initial_energy], [1000], linefmt='C4-', markerfmt=' ', basefmt=' ')
            energy_distribution.set_ylim(0, 1000)
        
        elif frame == 1: # Dirac delta at first observed momentum
            kde_energy.set_data([], [])  # Hide KDE
            _ = energy_distribution.stem([energies_np[1]], [1000], linefmt='C4-', markerfmt=' ', basefmt=' ')
            energy_distribution.set_ylim(0, 1000)

        elif frame > 1: # KDE for frame > 1 
            energy_subset = energies_np[1:frame+1]
            energy_density = expon.pdf(e_grid, scale=energy_subset.mean()) # Exponential reference distribution for positive energies
            energy_line.set_data(e_grid, energy_density) # * energy_subset.std() * len(energy_subset)**(-1/5)
            
            kde = gaussian_kde(energy_subset, bw_method=0.05)
            e_density = kde(e_grid)
            kde_energy.set_data(e_grid, e_density)
            energy_distribution.fill_between(e_grid, 0, e_density, color='C4', alpha=0.3) # Fill between for visibility
            
            max_density = max(e_density.max(), energy_density.max())
            energy_distribution.set_ylim(0, max_density * 1.1)

    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, hspace=0.4)
    plt.close(fig)
    
    animation = FuncAnimation(fig, _animate, frames=len(walk.data),
                        interval=walk.ms_between_frames, blit=False)
    
    if verbose:
        print("[BoundWalk] Rendering animation...")
    matplotlib.rcParams["animation.embed_limit"] = 50_000_000 # adjust (RC) for increasing animation file size to ~50 MB, may need to turn off for gif creation!
    display(HTML(animation.to_jshtml()))
    if verbose:
        print("[BoundWalk] Energy animation complete.")
