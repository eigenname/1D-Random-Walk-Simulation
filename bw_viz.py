"""
BoundWalk Visualization Module
Provides visualization functions for different observables from BoundWalk simulations
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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
    momenta_np = walk.data['nth Momentum'].to_numpy()
    times_np = walk.data['t'].to_numpy()
    entropies_np = walk.data['nth Entropy'].to_numpy()
    klds_np = walk.data['nth KLD'].to_numpy()
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 1], hspace=0.6)
    
    # ======================== Subplot 1: Particle Animation ========================
    ax_particle = fig.add_subplot(gs[0, 0])

    position_marker, = ax_particle.plot([], [], 'o', markersize=6, color='C0', label=r"$X_t$")
    momentum_vector = ax_particle.quiver([0], [0], angles='xy', scale_units='xy', scale=1, color='red', label=r"$P_t$")

    step_size = walk.step_size
    if isinstance(step_size, float): # display step size info in title, whether fixed 
        step_info = f"|ΔX| = {step_size}"
    else: # or randomly sampled from distributions
        if step_size['type'] == 'uniform':
            step_info = rf"|ΔX| \sim \mathcal{{U}}[{step_size['bounds'][0]}, {step_size['bounds'][1]}]"
        else:  # normal
            step_info = rf"|ΔX| \sim \mathcal{{N}}({step_size['params'][0]}, {step_size['params'][1]**2})"
    
    ax_particle.set_title(rf"$\mathcal{{BW}}: N={N}, {step_info}, X_0={init_position}$")
    ax_particle.get_yaxis().set_visible(False)
    ax_particle.set_xlabel(r"$X_t$")
    ax_particle.set_ylim(-0.05, 0.1)
    ax_particle.set_xlim(left_bound + momenta_np.min(), right_bound + momenta_np.max())
    ax_particle.vlines([left_bound, right_bound], ymin=-0.05, ymax=0.1, color='black', linewidth=0.7)
    ax_particle.legend(loc='upper left')
    
    # ======================== Subplot 2: Histogram ========================
    ax_hist = fig.add_subplot(gs[0, 1])
    x_grid = np.linspace(positions_np.min() - 0.1*positions_np.max(), positions_np.max() + 0.1*positions_np.max(), 1000)
    kde_positions, = ax_hist.plot([], [], color="C0", linewidth=1, label=r"KDE($X_t$)")
    ax_hist.axhline(1/(right_bound - left_bound), color="black", linestyle="--", linewidth=1, label=rf"$U[{left_bound},{right_bound}]$")
    
    ax_hist.set_title(r"$\rho(X_t)$ vs $X_t$")
    ax_hist.set_ylabel(r"$\rho(X_t)$")
    ax_hist.set_xlabel(r"$X_t$")
    ax_hist.set_xlim(x_grid.min(), x_grid.max())
    ax_hist.legend(loc='upper right')
    
    # ======================== Subplot 3: Position vs Time ========================
    ax_pos = fig.add_subplot(gs[1, :])
    line_pos, = ax_pos.plot([], [], color="C0", alpha=0.7)
    marker_pos, = ax_pos.plot([], [], ".", color="C0", label=r"$X_t$")
    
    ax_pos.set_title(r"$X_t$ vs $t$")
    ax_pos.set_ylabel(r"$X_t$")
    ax_pos.set_ylim(left_bound, right_bound)
    ax_pos.set_xlim(0, 10)
    ax_pos.tick_params(labelbottom=False)
    ax_pos.legend(loc='upper left')
    
    # ======================== Subplot 4: Entropy ========================
    ax_entr = fig.add_subplot(gs[2, :], sharex=ax_pos)
    line_entr, = ax_entr.plot([], [], color="C0", alpha=0.7)
    marker_entr, = ax_entr.plot([], [], ".", color="C0", label=r"$S_G[X_t]$")
    ax_entr.axhline(np.log(len(centers)), color='red', linewidth=0.7, linestyle='--', alpha=0.7, label=rf"$S_B = \ln({len(centers)})$")
    
    ax_entr.set_title(r"$S_G[X_t]$ vs $t$")
    ax_entr.set_ylabel(r"$S_G[X_t]$")
    ax_entr.set_ylim(0, np.log(len(centers)) + 0.5)
    ax_entr.tick_params(labelbottom=False)
    ax_entr.legend(loc='upper left')
    
    # ======================== Subplot 5: KL Divergence ========================
    ax_kld = fig.add_subplot(gs[3, :], sharex=ax_pos)
    line_kld, = ax_kld.plot([], [], color="C0", alpha=0.7)
    marker_kld, = ax_kld.plot([], [], ".", color="C0", label=r"$D_{KL}(\hat{\mathbb{P}}||U)$")
    
    ax_kld.set_title(r"$D_{KL}(\hat{\mathbb{P}}||U)$ vs $t$")
    ax_kld.set_ylabel(r"$D_{KL}$")
    ax_kld.set_xlabel(r"$t \to \infty$")
    ax_kld.set_ylim(0, np.log(len(centers)) + 0.5)
    ax_kld.legend(loc='upper left')
    
    # ======================== Animation Function ========================
    Δt = walk.time_scale
    milestones = {int(len(walk.data) * p) for p in [0.25, 0.5, 0.75, 1.0]} if verbose else set()
    
    def _animate(frame):
        if frame in milestones:
            print(f"[BoundWalk] Rendering: {frame}/{len(walk.data)} ({100*frame//len(walk.data)}%)")
        
        # Particle position and momentum
        position_marker.set_data([positions_np[frame]], [0])
        momentum_vector.set_offsets([[positions_np[frame], 0]])
        momentum_vector.set_UVC(momenta_np[frame], [0])
        
        if frame > 1:
            kde = gaussian_kde(positions_np[1:frame+1], bw_method=0.1 * positions_np[1:frame+1].std() * len(positions_np[1:frame+1])**(-1/5))
            x_density = kde(x_grid)
            kde_positions.set_data(x_grid, x_density)
            ax_hist.set_ylim(0, x_density.max() * 1.1)

        # Position trajectory
        line_pos.set_data(times_np[:frame+1], positions_np[:frame+1])
        marker_pos.set_data([times_np[frame]], [positions_np[frame]])
        
        x_max = times_np[frame] if frame > 2 else 2
        ax_pos.set_xlim(0, x_max)
        ax_pos.xaxis.set_major_locator(MaxNLocator(integer=False, prune='both', nbins=6))
        
        # Entropy
        if frame > 0:
            line_entr.set_data(times_np[1:frame+1], entropies_np[1:frame+1])
            marker_entr.set_data([times_np[frame]], [entropies_np[frame]])
        
        # KLD
        if frame > 0:
            line_kld.set_data(times_np[1:frame+1], klds_np[1:frame+1])
            marker_kld.set_data([times_np[frame]], [klds_np[frame]])
    
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
    N = walk.total_steps
    momenta_np = walk.data['nth Momentum'].to_numpy()
    times_np = walk.data['t'].to_numpy()
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.6)
    
    # ======================== Subplot 1: Momentum Distribution ========================
    ax_hist = fig.add_subplot(gs[0])
    p_min, p_max = momenta_np.min(), momenta_np.max()
    p_grid = np.linspace(momenta_np.min() + 0.1*momenta_np.min(), momenta_np.max() + 0.1*momenta_np.max(), 1000)

    kde_momenta, = ax_hist.plot([], [], color="red", lw=1, label=r'KDE($P_t$)')
    ax_hist.set_title(r"$\rho(P_t)$ vs $P_t$")
    ax_hist.set_ylabel(r"$\rho(P_t)$")
    ax_hist.set_xlabel(r"$P_t$")
    ax_hist.set_xlim(p_grid.min(), p_grid.max())

    # ======================== Subplot 2: Momentum Trajectory ========================
    ax_traj = fig.add_subplot(gs[1])
    line_traj, = ax_traj.plot([], [], color="red", alpha=0.7)
    marker_traj, = ax_traj.plot([], [], ".", color="red", label=r"$P_t$")

    ax_traj.set_title(r"$P_t$ vs $t$")
    ax_traj.set_ylabel(r"$P_t$")
    ax_traj.set_xlabel(r"$t \to \infty$")
    ax_traj.set_xlim(0, 10)
    ax_traj.legend(loc='upper left')
    
    # ======================== Subplot 3: Phase Space (X, P) ========================
    ax_phase = fig.add_subplot(gs[2])
    positions_np = walk.data['nth Position'].to_numpy()  
    phase_point, = ax_phase.plot([], [], ".", color="purple")
    phase_traj, = ax_phase.plot([], [], color="purple", alpha=0.25)
    
    ax_phase.hlines(p_max, xmin=positions_np.min(), xmax=positions_np.max(), color='red', linewidth=0.7, linestyle='--')
    ax_phase.hlines(p_min, xmin=positions_np.min(), xmax=positions_np.max(), color='red', linewidth=0.7, linestyle='--')
    ax_phase.vlines(positions_np.max(), ymin=p_min, ymax=p_max, color='blue', linewidth=0.7, linestyle='--')
    ax_phase.vlines(positions_np.min(), ymin=p_min, ymax=p_max, color='blue', linewidth=0.7, linestyle='--')

    ax_phase.set_title(r"Phase Space: $(X_t, P_t)$")
    ax_phase.set_xlabel(r"$X_t$")
    ax_phase.set_ylabel(r"$P_t$")
    ax_phase.set_xlim(positions_np.min() - 0.1*positions_np.max(), positions_np.max() + 0.1*positions_np.max())
    
    # ======================== Animation Function ========================
    milestones = {int(len(walk.data) * p) for p in [0.25, 0.5, 0.75, 1.0]} if verbose else set()
    
    def _animate(frame):
        if frame in milestones:
            print(f"[BoundWalk] Rendering: {frame}/{len(walk.data)} ({100*frame//len(walk.data)}%)")
        
        # Momentum distribution
        if frame > 1:
            kde = gaussian_kde(momenta_np[1:frame+1], bw_method=0.1 * momenta_np[1:frame+1].std() * len(momenta_np[1:frame+1])**(-1/5))
            p_density = kde(p_grid)
            kde_momenta.set_data(p_grid, p_density)
            ax_hist.set_ylim(0, p_density.max() * 1.1)

        # Momentum trajectory
        line_traj.set_data(times_np[:frame+1], momenta_np[:frame+1])
        marker_traj.set_data([times_np[frame]], [momenta_np[frame]])
        ymin = momenta_np[:frame+1].min()
        ymax = momenta_np[:frame+1].max()
        if ymin == ymax:
            pad = 1e-6 if ymin == 0 else abs(ymin) * 0.1
        else:
            pad = 0.1 * (ymax - ymin)

        ax_traj.set_ylim(ymin - pad, ymax + pad)

        x_max = times_np[frame] if frame > 2 else 2
        ax_traj.set_xlim(0, x_max)

        # Phase space
        if frame > 0:
            phase_point.set_data(positions_np[:frame+1], momenta_np[:frame+1])
            phase_traj.set_data(positions_np[:frame+1], momenta_np[:frame+1])
            
            ax_phase.set_ylim(momenta_np[:frame+1].min() * 1.1,momenta_np[:frame+1].max() * 1.1)
    
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
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
    
    # ======================== Subplot 1: Energy vs Time ========================
    ax_traj = fig.add_subplot(gs[0, :])
    line_traj, = ax_traj.plot([], [], color="C2", alpha=0.7)
    marker_traj, = ax_traj.plot([], [], ".", color="C2", label=r"$E_t$")
    
    # Mark theoretical energy if fixed step
    if isinstance(walk.step_size, float):
        E_theory = 0.5 * (walk.step_size / walk.time_scale)**2
        ax_traj.axhline(E_theory, color='red', linewidth=0.7, linestyle='--', 
                       alpha=0.7, label=f"$E_{{theory}} = {E_theory:.6f}$")
    
    ax_traj.set_title(r"$E_t = \frac{P_t^2}{2m}$ vs $t$")
    ax_traj.set_ylabel(r"$E_t$")
    ax_traj.set_xlim(0, 10)
    ax_traj.tick_params(labelbottom=False)
    ax_traj.legend(loc='upper right')
    
    # ======================== Subplot 2: Energy Distribution ========================
    ax_hist = fig.add_subplot(gs[1, 0])
    e_min, e_max = energies_np[1:].min(), energies_np[1:].max()
    e_range = e_max - e_min if e_max != e_min else 1e-6
    bins = np.linspace(max(0, e_min - 0.1*e_range), e_max + 0.1*e_range, 50)
    
    hist_counts, hist_edges = np.histogram([], bins=bins)
    hist_width = hist_edges[1] - hist_edges[0]
    bars = ax_hist.bar(hist_edges[:-1], hist_counts, width=hist_width,
                       align='edge', color="C2", edgecolor="black", alpha=0.7)
    
    ax_hist.set_title(r"$\mathbb{P}(E_t)$ Distribution")
    ax_hist.set_ylabel("Probability Density")
    ax_hist.set_xlabel(r"$E_t$")
    
    # ======================== Subplot 3: Energy Statistics ========================
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.axis('off')
    stat_text = ax_stats.text(0.1, 0.9, "", transform=ax_stats.transAxes,
                              verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    # ======================== Subplot 4: Phase Space with Energy Contours ========================
    ax_phase = fig.add_subplot(gs[2, :])
    scatter = ax_phase.scatter([], [], c=[], cmap='plasma', alpha=0.6, s=15)
    
    # Energy contours (for visualization)
    if isinstance(walk.step_size, float):
        x_grid = np.linspace(*walk.boundaries, 100)
        for E_level in [E_theory * 0.5, E_theory, E_theory * 1.5]:
            p_level = np.sqrt(2 * E_level)
            ax_phase.axhline(p_level, color='gray', linewidth=0.5, alpha=0.3, linestyle=':')
            ax_phase.axhline(-p_level, color='gray', linewidth=0.5, alpha=0.3, linestyle=':')
    
    ax_phase.set_title(r"Phase Space Colored by Energy")
    ax_phase.set_xlabel(r"$X_t$")
    ax_phase.set_ylabel(r"$P_t$")
    ax_phase.set_xlim(walk.boundaries)
    
    # ======================== Animation Function ========================
    milestones = {int(len(walk.data) * p) for p in [0.25, 0.5, 0.75, 1.0]} if verbose else set()
    
    def _animate(frame):
        if frame in milestones:
            print(f"[BoundWalk] Rendering: {frame}/{len(walk.data)} ({100*frame//len(walk.data)}%)")
        
        # Energy trajectory
        line_traj.set_data(times_np[:frame+1], energies_np[:frame+1])
        marker_traj.set_data([times_np[frame]], [energies_np[frame]])
        
        x_max = times_np[frame] if frame > 2 else 2
        ax_traj.set_xlim(0, x_max)
        if frame > 0:
            e_min_current = energies_np[1:frame+1].min()
            e_max_current = energies_np[1:frame+1].max()
            ax_traj.set_ylim(e_min_current * 0.9, e_max_current * 1.1)
        
        # Energy distribution
        if frame > 10:
            counts, _ = np.histogram(energies_np[1:frame+1], bins=bins)
            probs = counts / counts.sum() if counts.sum() > 0 else counts
            
            for bar, height in zip(bars, probs):
                bar.set_height(height)
            
            ax_hist.set_ylim(0, probs.max() * 1.1 if probs.max() > 0 else 1)
        
        # Statistics text
        if frame > 0:
            e_current = energies_np[1:frame+1]
            stats_str = (
                f"Steps: {frame}/{N}\n"
                f"E_mean: {e_current.mean():.8f}\n"
                f"E_std:  {e_current.std():.8f}\n"
                f"E_min:  {e_current.min():.8f}\n"
                f"E_max:  {e_current.max():.8f}"
            )
            stat_text.set_text(stats_str)
        
        # Phase space
        if frame > 0:
            scatter.set_offsets(np.c_[positions_np[:frame+1], momenta_np[:frame+1]])
            scatter.set_array(energies_np[:frame+1])
            
            p_range = max(abs(momenta_np[:frame+1].min()), abs(momenta_np[:frame+1].max()))
            ax_phase.set_ylim(-p_range * 1.1, p_range * 1.1)
    
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, hspace=0.4)
    plt.close(fig)
    
    anim = FuncAnimation(fig, _animate, frames=len(walk.data),
                        interval=walk.ms_between_frames, blit=False)
    
    if verbose:
        print("[BoundWalk] Rendering animation...")
    display(anim)
    if verbose:
        print("[BoundWalk] Energy animation complete.")
