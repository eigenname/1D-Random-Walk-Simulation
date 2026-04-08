from bw_tools import ( # import all helper functions from bw_tools.py
    find_rounding_precision,
    create_displacements,
    pad_to_domain,
    reflect,
    get_bin_width
)

import numpy as np # initially, to account for seed reproducibility
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd # for storing generated data in a structured format
import matplotlib.pyplot as plt # initially, to enable file format for displaying anim inline
from matplotlib.gridspec import GridSpec # to encapsulate all plots proportionally
from matplotlib import rcParams # for runtime configuration of animation rendering
rcParams["animation.html"] = "jshtml" # render animation as interactive HTML widget inline
rcParams["animation.embed_limit"] = 50_000_000 # increases animation file size to 50 MB
from matplotlib.animation import FuncAnimation # for creating the animation
from matplotlib.ticker import AutoLocator, MaxNLocator # for integer ticks/labels for entropy vs n plot

from dataclasses import dataclass # for cleaner class definition
from collections import defaultdict # dict that keeps occurrence count for each position bin
from scipy.stats import entropy # actually KLD method
from IPython.display import HTML, display # to display anim inline


@dataclass(kw_only=True) # use keyword-only arguments for clarity and to avoid confusion when instantiating the class with many parameters
class BoundWalk:
    total_steps: int # number of steps in the random walk, denoted as N
    step_size: tuple | (float | int) # step size can be randomly sampled (uniform or standard normal) or fixed (float or int)
    time_scale: float = 1.0 # default time scale for the walk
    boundaries: tuple = (0, 1) # default boundaries for the walk

    mass: float = 1.0 # default mass of the particle
    initial_position: float = 0 # default initial position at 0
    initial_velocity: float = 0 # default initial velocity at 0

    seed: int = None # optional seed for reproducibility, default is 'None'.
    ms_between_frames: int = 500 # milliseconds between frames in animation, default is 500ms (0.5s)
    #__________________________________________________________________________________________________________________________
    def __post_init__(self): 
        self.__simulate__()
    #__________________________________________________________________________________________________________________________
    def __simulate__(self): # generates data for random walk, simulated based on given parameters
        Δt = self.time_scale
        m = self.mass
        X0 = self.initial_position
        V0 = self.initial_velocity
        P0 = m * V0 / Δt 
        E0 = P0**2 / (2*m)
        seed = self.seed
        N = self.total_steps
        ΔX = self.step_size
        rounding_precision = find_rounding_precision(ΔX) # determine rounding precision based on step_size, whether to be fixed or randomly sampled
        self.displacements = create_displacements(seed, N, ΔX, Δt) # create displacements based on given parameters, whether fixed or randomly sampled
        left_bound, right_bound = self.boundaries

        Positions = [X0] # initialize positions list with initial position X0
        Velocities = [V0] # initialize velocities list with initial velocity V0
        Momenta = [P0] # initialize momenta list with initial momentum P0
        Energies = [E0] # initialize energies list with initial energy E0
        counts = defaultdict(int) # dict that keeps count of probs per outcome
        Possible_Outcomes = [np.array([X0])]   # particle is at X0 with certainty
        Probabilities     = [np.array([1.0])]        # p(X0) = 1
        Entropies         = [0.0]                    # H = 0, certain state
        KLDs             = [0.0]                    # KLD = 0, nothing has diverged yet
 
        self.bin_width = get_bin_width(ΔX) # determine bin width for histogram and domain definition, based on step_size parameters, whether to be fixed or randomly sampled 
        self.domain = np.arange(left_bound, right_bound + 1e-8, self.bin_width) 
        uniform_probs = np.ones(len(self.domain)) / len(self.domain) # needed for computing KLD

        Displacements = [0] # to record displacements after corrections within reflective bounds [0,1]
        for step in self.displacements:
            original_step = Positions[-1] + step
            next_position = reflect(original_step, left_bound, right_bound)   
            next_position = round(next_position, rounding_precision)
            step = round(next_position - Positions[-1], rounding_precision)  
            Positions.append(next_position) 
            Displacements.append(step) # compute actual displacement after refleciton, then append
            Velocities.append(step / Δt) # compute and append next velocity
            Momenta.append(m * Velocities[-1]) # compute and append next momentum
            Energies.append(round(Momenta[-1]**2 / (2*m), rounding_precision) if isinstance(ΔX, dict) else Momenta[-1]**2 / (2*m)) # compute and append next energy, apply rounding when sampling

            counts[next_position] += 1
            total = sum(counts.values())
            outcomes = np.array(list(counts.keys()))
            probabilities = np.array([counts[o] / total for o in outcomes])
            H = -(probabilities * np.log(probabilities)).sum() # compute entropy using the probabilities of the outcomes at this step
            Possible_Outcomes.append(outcomes)
            Probabilities.append(probabilities)
            Entropies.append(H)

            empirical_probs = pad_to_domain(outcomes, probabilities, self.domain, left_bound, self.bin_width) # align the outcomes and probabilities from each step with the defined domain, to get empirical distribution in the same support as uniform distribution for KLD computation
            kld = entropy(empirical_probs, uniform_probs) # compute KLD
            KLDs.append(kld)

        self.data = pd.DataFrame({ # tabular data of simulation
            "t": np.arange(N+1) * Δt,
            "n ≤ N": np.arange(N+1),
            "nth Position": Positions,
            "nth Displacement": Displacements,
            "nth Velocity": Velocities,
            "nth Momentum": Momenta,
            "nth Energy": Energies,
            "nth Possible Outcomes": Possible_Outcomes,
            "nth Probabilities": Probabilities,
            "nth Entropy": Entropies,
            "nth KLD": KLDs
        })
    #__________________________________________________________________________________________________________________________
    def __visualize__(self): 
        X0 = self.initial_position
        N = self.total_steps
        ΔX = self.step_size
        left_bound, right_bound = self.boundaries
        bin_width = self.bin_width
        centers = self.domain 
        edges = np.append(centers - bin_width/2, centers[-1] + bin_width/2)

        fig = plt.figure(figsize=(12,8))
        gs = GridSpec(4, 2, height_ratios=[1,1,1,1], hspace=0.6)

        #----------------------- particle_animation ---------------------- !!! 1st row, 1st col: 1D Position(N) !!!
        particle_animation = fig.add_subplot(gs[0,0]) # 1d bound rand walk anim
        position, = particle_animation.plot([], [], 'o', markersize=6, color='C0', label="particle") # initialize scatter marker for anim
        particle_animation.axhline(0, color='black', linewidth=0.7, alpha=0.3) # reference line for y=0

        if isinstance(ΔX, (float, int)):
            step_info = f"|ΔX| = {ΔX}"
        else:
            match self.step_size['type']:
                case 'uniform':
                    step_info = rf"|ΔX| \sim \mathcal{{U}}[{ΔX['bounds'][0]}, {ΔX['bounds'][1]}]"
                case 'normal':
                    step_info = rf"|ΔX| \sim \mathcal{{N}}({ΔX['params'][0]}, {ΔX['params'][1]})"
        
        particle_animation.set_title(rf"$\mathcal{{BW}}(N={{{N}}}, {step_info}; X_0 \equiv {{{X0}}}): \ x_{{_{{t}}}} \in \mathbb{{R}}_{{_{{{[left_bound, right_bound]}}}}}$")

        particle_animation.get_yaxis().set_visible(False) # don't need to see yaxis ticks/labels
        particle_animation.set_xlabel(rf"$X_{{_{{t}}}} = x_{{_{{t}}}} \in \mathbb{{R}}_{{_{{{[left_bound, right_bound]}}}}}$")
        particle_animation.set_ylim(-0.05, 0.1) # limit yaxis dimensions
        particle_animation.set_xlim(left_bound, right_bound) # walk will be confined within (a,b)
        particle_animation.legend(loc='upper left')

        #----------------------- histogram ---------------------- !!! 1st row, 2nd col: Probability Histogram !!!
        histogram = fig.add_subplot(gs[0,1]) # prob hist of positions
        bars = histogram.bar(centers, np.zeros_like(centers), width=bin_width, align='center',color="C0", edgecolor="black", alpha=0.7)
        histogram.axhline(1/len(centers), color="red", linestyle="--", linewidth=0.8, label=rf"$U[{left_bound},{right_bound}]$")

        histogram.set_title(r"$\mathbb{P}(X_{{_{{t}}}} = x_{{_{{t}}}})$ Histogram")
        histogram.set_ylabel(r"$\mathbb{P} \in \mathbb{{R}}_{{_{[0, 1]}}}$ ")
        histogram.set_xlabel(rf"$X_{{_{{t}}}} = x_{{_{{t}}}} \in \mathbb{{R}}_{{_{{{[left_bound, right_bound]}}}}}$")
        histogram.set_xlim(left_bound, right_bound)
        histogram.legend(loc='upper right') # enables label for U(a,b) PDF, fix to upper right

        #----------------------- position_plot ---------------------- !!! 2nd row: Position vs N plot !!!
        position_plot = fig.add_subplot(gs[1, :]) # plot of positions vs n
        line_plot, = position_plot.plot([], [], color="C0", alpha=0.7)
        marker_plot, = position_plot.plot([], [], ".", color="C0", label=r"$x_{{_{{t}}}}$")

        position_plot.set_title(r"$X_{{_{{t}}}} = x_{{_{{t}}}}$ vs $t \to N$")
        position_plot.set_ylabel(r"$X_{{_{{t}}}} = x_{{_{{t}}}}$")
        position_plot.set_ylim(left_bound, right_bound) # limit yaxis dimensions to boundaries
        position_plot.set_xlim(0, 10)    # add this — propagates to ax_entr and ax_norm via sharex
        position_plot.tick_params(labelbottom=False)   # hide x tick labels
        position_plot.legend(loc='upper left')

        #----------------------- entropy_plot ---------------------- !!! 3rd row: Entropy vs N Plot !!!
        entropy_plot = fig.add_subplot(gs[2,:], sharex=position_plot) # entropy vs n
        line_entr, = entropy_plot.plot([], [], color="C0", alpha=0.7)
        marker_entr, = entropy_plot.plot([], [], ".", color="C0", label=rf"$H[X_{{_{{t}}}}]$")
        entropy_plot.axhline(np.log(len(centers)), color='red', linewidth=0.7, linestyle='--', alpha=0.7, label=rf"$\ln({len(centers) + 1})$")  # Boltzmann Entropy supremum

        entropy_plot.set_title(r"$H[X_{{_{{t}}}}]$ vs $t \to N$")
        entropy_plot.set_ylabel(r"$H[X_{{_{{t}}}}]$")
        entropy_plot.set_ylim(0, np.log(len(centers))+0.1) # default before any data — positive only
        entropy_plot.tick_params(labelbottom=False)   # hide x tick labels
        entropy_plot.legend(loc='upper left')

        #----------------------- kld_plot ---------------------- !!! 4th row: KLD vs N Plot !!!
        kld_plot = fig.add_subplot(gs[3,:], sharex=position_plot) # KLD vs N
        line_kld, = kld_plot.plot([], [], color="C0", alpha=0.7)
        marker_kld, = kld_plot.plot([], [], ".", color="C0", label=rf"$D_{{KL}}(\mathbb{{P}}||U)$")

        kld_plot.set_title(r"$D_{{KL}}(\mathbb{P}||U)$ vs $t \to \infty$")
        kld_plot.set_ylabel(r"$D_{{KL}}(\mathbb{P}||U)$")
        kld_plot.set_xlabel(r"$t \to \infty$")
        kld_plot.set_ylim(0, np.log(len(centers)) + 0.1)      # default before any data — positive only 
        kld_plot.legend(loc='upper left')

        #=========================================================================================== # CHANGE name to something better!
        def _animate(frame): # i within [0, len(walk.data)]
            Δt = self.time_scale
            #----------------------- particle_animation ---------------------- !!! 1st row, 1st col: 1D Position(N) !!!
            current_pos = self.data['nth Position'].iloc[frame] # find nth Position given t=frame
            position.set_data([current_pos], [0]) # move position marker to (x=current_pos, y=0)

            #----------------------- histogram ---------------------- !!! 1st row, 2nd col: Probability Histogram !!!
            if frame == 0: # delta distribution at initial position
                probs = np.zeros(len(bars))
                x0_idx = round((self.initial_position - left_bound) / bin_width)
                probs[x0_idx] = 1.0

            else: # compute histogram probabilities based on positions up to current frame
                current_positions = self.data['nth Position'].iloc[1:frame+1]
                counts, _ = np.histogram(current_positions, bins=edges)
                probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)

            for bar, height in zip(bars, probs): # update histogram bars
                bar.set_height(height)

            histogram.xaxis.set_major_locator(AutoLocator())
            histogram.set_ylim(0, probs.max() * 1.05)
            yticks = np.linspace(0, probs.max(), 5)
            histogram.set_yticks(yticks)
            histogram.set_yticklabels([f"{y:.3f}" for y in yticks])

            #----------------------- position_plot ---------------------- !!! 2nd row: Position vs N plot !!!
            steps = self.data['t'].iloc[:frame+1] # steps from 0 to current frame n, for x-axis of position plot 
            positions = self.data['nth Position'].iloc[:frame+1]
            line_plot.set_data(steps, positions)
            marker_plot.set_data([frame * Δt], [self.data['nth Position'].iloc[frame]]) # move marker to current position at this frame, adjusted by time scale

            x_max = frame * Δt if frame > 2 else 2
            position_plot.relim()
            position_plot.set_xlim(0, x_max)   # expand as n grows beyond 10
            position_plot.xaxis.set_major_locator(MaxNLocator(integer=False, prune='both', nbins=6)) # set x ticks at multiples of Δt for shared plots

            #----------------------- entropy_plot ---------------------- !!! 3rd row: Entropy vs N Plot !!!
            steps = self.data['t'].iloc[1:frame+1] # steps from 1 to current frame n, for x-axis of entropy and KLD plots
            entropies = self.data['nth Entropy'].iloc[1:frame+1]
            line_entr.set_data(steps, entropies)
            marker_entr.set_data([frame * Δt], [self.data['nth Entropy'].iloc[frame]])
            entropy_plot.relim()
            entropy_plot.xaxis.set_major_locator(MaxNLocator(integer=False, prune='both', nbins=6))

            #----------------------- kld_plot ---------------------- !!! 4th row: KLD vs N Plot !!!
            klds = self.data['nth KLD'].iloc[1:frame+1]
            line_kld.set_data(steps, klds)
            marker_kld.set_data([frame * Δt], [self.data['nth KLD'].iloc[frame]])
            kld_plot.relim()
            kld_plot.xaxis.set_major_locator(MaxNLocator(integer=False, prune='both', nbins=6))
        #===========================================================================================
        plt.subplots_adjust(left=0.075, bottom=0.075, hspace=0.4)  # for adjusting margins
        plt.close(fig) # ensure no static plots are displayed
        anim = FuncAnimation(fig, _animate, frames=len(self.data),
                interval=self.ms_between_frames, # delay between frames in milliseconds 
                blit=False  # blitting doesn’t play well with clearing/replotting
            )

        display(HTML(anim.to_jshtml())) # for interactive HTML widget inline in notebook
    #__________________________________________________________________________________________________________________________