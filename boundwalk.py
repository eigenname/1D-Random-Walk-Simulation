from bw_tools import ( # import all helper functions from bw_tools.py
    find_rounding_precision,
    create_displacements,
    define_domain,
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
from matplotlib.ticker import AutoLocator # for integer ticks/labels for entropy vs n plot

from dataclasses import dataclass # for cleaner class definition
from collections import defaultdict # dict that keeps occurrence count for each position bin
from scipy.stats import entropy # actually KLD method
from IPython.display import HTML, display # to display anim inline


@dataclass(kw_only=True) # use keyword-only arguments for clarity and to avoid confusion when instantiating the class with many parameters
class BoundWalk:
    total_steps: int # number of steps in the random walk, denoted as N
    step_size: tuple | (float | int) # step size can be randomly sampled (uniform or standard normal) or fixed (float or int)
    initial_position: float = 0 # default initial position at 0
    boundaries: tuple = (0, 1) # default boundaries for the walk
    seed: int = None # optional seed for reproducibility, default is 'None'.
    fps: int = 500 # frames per second for animation, adjust as needed for smoother or faster animation
    #__________________________________________________________________________________________________________________________
    def __post_init__(self): 
        self.__simulate__()
        self.__visualize__()
    #__________________________________________________________________________________________________________________________
    def __simulate__(self): # generates data for random walk, simulated based on given parameters
        X0 = self.initial_position
        seed = self.seed
        N = self.total_steps
        ΔX = self.step_size
        rounding_precision = find_rounding_precision(ΔX) # determine rounding precision based on step_size, whether to be fixed or randomly sampled
        self.displacements = create_displacements(seed, N, ΔX) # create displacements based on given parameters, whether fixed or randomly sampled
        left_bound, right_bound = self.boundaries

        Positions = [X0] # initialize positions list with initial position X0
        counts = defaultdict(int) # dict that keeps count of probs per outcome
        Possible_Outcomes = [np.array([X0])]   # particle is at X0 with certainty
        Probabilities     = [np.array([1.0])]        # p(X0) = 1
        Entropies         = [0.0]                    # H = 0, certain state
        KLDs             = [0.0]                    # KLD = 0, nothing has diverged yet
 
        self.bin_width = get_bin_width(ΔX) # if step_size is float/int, is fixed, bin width is just the step size itself
        self.domain = define_domain(left_bound, right_bound, ΔX) 
        uniform_probs = np.ones(len(self.domain)) / len(self.domain) # needed for computing KLD

        Displacements = [0] # to record displacements after corrections within reflective bounds [0,1]
        for step in self.displacements:
            original_step = Positions[-1] + step
            next_position = reflect(original_step, left_bound, right_bound)   
            next_position = round(next_position, rounding_precision)
            step = round(next_position - Positions[-1], rounding_precision)  
            Positions.append(next_position) 
            Displacements.append(step) # compute actual displacement after refleciton, then append

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
            "n ≤ N": np.arange(N+1),
            "nth Displacement": Displacements,
            "nth Position": Positions,
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
        eigenvalues = self.domain # 
        edges = np.append(eigenvalues - bin_width/2, eigenvalues[-1] + bin_width/2)

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
        
        particle_animation.set_title(rf"$\mathcal{{BW}}(N={{{N}}}, {step_info}; X_0 \equiv {{{X0}}}): \ x_{{_{{n}}}} \in \mathbb{{R}}_{{_{{{[left_bound, right_bound]}}}}}$")

        particle_animation.get_yaxis().set_visible(False) # don't need to see yaxis ticks/labels
        particle_animation.set_xlabel(rf"$X_{{_{{n}}}} = x_{{_{{n}}}} \in \mathbb{{R}}_{{_{{{[left_bound, right_bound]}}}}}$")
        particle_animation.set_ylim(-0.05, 0.1) # limit yaxis dimensions
        particle_animation.set_xlim(left_bound, right_bound) # walk will be confined within (a,b)
        particle_animation.legend(loc='upper left')

        #----------------------- histogram ---------------------- !!! 1st row, 2nd col: Probability Histogram !!!
        histogram = fig.add_subplot(gs[0,1]) # prob hist of positions
        bars = histogram.bar(eigenvalues, np.zeros_like(eigenvalues), width=bin_width, align='center',color="C0", edgecolor="black", alpha=0.7)
        histogram.axhline(1/len(eigenvalues), color="red", linestyle="--", linewidth=0.8, label=rf"$U[{left_bound},{right_bound}]$")

        histogram.set_title(r"$\mathbb{P}(X_{{_{{n}}}} = x_{{_{{n}}}})$ Histogram")
        histogram.set_ylabel(r"$\mathbb{P} \in \mathbb{{R}}_{{_{[0, 1]}}}$ ")
        histogram.set_xlabel(rf"$X_{{_{{n}}}} = x_{{_{{n}}}} \in \mathbb{{R}}_{{_{{{[left_bound, right_bound]}}}}}$")
        histogram.set_xlim(left_bound, right_bound)
        histogram.legend(loc='upper right') # enables label for U(a,b) PDF, fix to upper right

        #----------------------- position_plot ---------------------- !!! 2nd row: Position vs N plot !!!
        position_plot = fig.add_subplot(gs[1, :]) # plot of positions vs n
        line_plot, = position_plot.plot([], [], color="C0", alpha=0.7)
        marker_plot, = position_plot.plot([], [], ".", color="C0", label=r"$x_{{_{{n}}}}$")

        position_plot.set_title(r"$X_{{_{{n}}}} = x_{{_{{n}}}}$ vs $n \to N$")
        position_plot.set_ylabel(r"$X_{{_{{n}}}} = x_{{_{{n}}}}$")
        position_plot.set_ylim(left_bound, right_bound) # limit yaxis dimensions to boundaries
        position_plot.set_xlim(0, 10)    # add this — propagates to ax_entr and ax_norm via sharex
        position_plot.tick_params(labelbottom=False)   # hide x tick labels
        position_plot.legend(loc='upper left')

        #----------------------- entropy_plot ---------------------- !!! 3rd row: Entropy vs N Plot !!!
        entropy_plot = fig.add_subplot(gs[2,:], sharex=position_plot) # entropy vs n
        line_entr, = entropy_plot.plot([], [], color="C0", alpha=0.7)
        marker_entr, = entropy_plot.plot([], [], ".", color="C0", label=rf"$H[X_{{_{{n}}}}]$")
        entropy_plot.axhline(np.log(len(eigenvalues)), color='red', linewidth=0.7, linestyle='--', alpha=0.7, label=rf"$\ln({len(eigenvalues)})$")  # Boltzmann Entropy supremum

        entropy_plot.set_title(r"$H[X_{{_{{n}}}}]$ vs $n \to N$")
        entropy_plot.set_ylabel(r"$H[X_{{_{{n}}}}]$")
        entropy_plot.set_ylim(0, np.log(len(eigenvalues))+0.1) # default before any data — positive only
        entropy_plot.tick_params(labelbottom=False)   # hide x tick labels
        entropy_plot.legend(loc='upper left')

        #----------------------- kld_plot ---------------------- !!! 4th row: KLD vs N Plot !!!
        kld_plot = fig.add_subplot(gs[3,:], sharex=position_plot) # KLD vs N
        line_kld, = kld_plot.plot([], [], color="C0", alpha=0.7)
        marker_kld, = kld_plot.plot([], [], ".", color="C0", label=rf"$D_{{KL}}(\mathbb{{P}}||U)$")

        kld_plot.set_title(r"$D_{{KL}}(\mathbb{P}||U)$ vs $n \to N$")
        kld_plot.set_ylabel(r"$D_{{KL}}(\mathbb{P}||U)$")
        kld_plot.set_xlabel(r"$n \to N$")
        kld_plot.set_ylim(0, np.log(len(eigenvalues)) + 0.1)      # default before any data — positive only 
        kld_plot.legend(loc='upper left')

        #=========================================================================================== # CHANGE name to something better!
        def _animate(frame): # i within [0, len(walk.data)]
            #----------------------- particle_animation ---------------------- !!! 1st row, 1st col: 1D Position(N) !!!
            current_pos = self.data["nth Position"].iloc[frame] # find nth Position given n=frame
            position.set_data([current_pos], [0]) # move position marker to (x=current_pos, y=0)

            #----------------------- histogram ---------------------- !!! 1st row, 2nd col: Probability Histogram !!!
            if frame == 0: # delta distribution at initial position
                probs = np.zeros(len(bars))
                x0_idx = round((self.initial_position - left_bound) / bin_width)
                probs[x0_idx] = 1.0

            else: # compute histogram probabilities based on positions up to current frame
                current_positions = self.data["nth Position"].iloc[1:frame+1]
                counts, _ = np.histogram(current_positions, bins=edges)
                probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)

            for bar, height in zip(bars, probs): # update histogram bars
                bar.set_height(height)

            histogram.xaxis.set_major_locator(AutoLocator())
            max_prob = probs.max()
            if max_prob > 0: # dynamic y-axis scaling based on max prob at current frame
                histogram.set_ylim(0, max_prob * 1.1)
                yticks = np.linspace(0, max_prob, 5)
                histogram.set_yticks(yticks)
                histogram.set_yticklabels([f"{y:.3f}" for y in yticks])

            else: # initial frame with delta distribution, max_prob is 1, set static y-axis
                histogram.set_ylim(0, 1.05)
                histogram.set_yticks([0.0, 0.25, 0.5, 0.75, 1.00])
                histogram.set_yticklabels(["0.000", "0.250", "0.500", "0.750", "1.000"])

            #----------------------- position_plot ---------------------- !!! 2nd row: Position vs N plot !!!
            steps = self.data['n ≤ N'].iloc[:frame+1] # steps from 0 to current frame n, for x-axis of position plot 
            positions = self.data["nth Position"].iloc[:frame+1]
            line_plot.set_data(steps, positions)
            marker_plot.set_data([self.data['n ≤ N'].iloc[frame]], [self.data['nth Position'].iloc[frame]])
            position_plot.relim()
            if frame <= 5:
                position_plot.set_xlim(0, 5)      # fixed at [0,10] for first 10 frames

            else:
                position_plot.set_xlim(0, frame)   # expand as n grows beyond 10
            position_plot.xaxis.set_major_locator(AutoLocator()) # only integer x ticks!

            #----------------------- entropy_plot ---------------------- !!! 3rd row: Entropy vs N Plot !!!
            steps = self.data['n ≤ N'].iloc[1:frame+1] # steps from 1 to current frame n, for x-axis of entropy and KLD plots
            entropies = self.data['nth Entropy'].iloc[1:frame+1]
            line_entr.set_data(steps, entropies)
            marker_entr.set_data([self.data['n ≤ N'].iloc[frame]], [self.data['nth Entropy'].iloc[frame]])
            entropy_plot.relim()
            entropy_plot.xaxis.set_major_locator(AutoLocator()) # only integer x ticks!

            #----------------------- kld_plot ---------------------- !!! 4th row: KLD vs N Plot !!!
            klds = self.data['nth KLD'].iloc[1:frame+1]
            line_kld.set_data(steps, klds)
            marker_kld.set_data([self.data['n ≤ N'].iloc[frame]], [self.data['nth KLD'].iloc[frame]])
            kld_plot.relim()
            kld_plot.xaxis.set_major_locator(AutoLocator()) # only integer x ticks!
        #===========================================================================================
        plt.subplots_adjust(left=0.075, bottom=0.075, hspace=0.4)  # for adjusting margins
        plt.close(fig) # ensure no static plots are displayed
        anim = FuncAnimation(fig, _animate, frames=len(self.data),
                interval=self.fps, # delay between frames in milliseconds 
                blit=False  # blitting doesn’t play well with clearing/replotting
            )

        display(HTML(anim.to_jshtml())) # for interactive HTML widget inline in notebook
    #__________________________________________________________________________________________________________________________