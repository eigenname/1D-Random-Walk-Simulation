import numpy as np # initially, to account for seed reproducibility
import pandas as pd # to store all generated data for reference
import math # for computing readable tick steps
from collections import defaultdict # essentially, dict that keeps occurrence count for each position bin

import matplotlib.pyplot as plt # initially, to enable file format for displaying anim inline
import matplotlib # expand limit for data capacity of generated anim
import matplotlib.animation as animation 
import matplotlib.gridspec as gridspec # to encapsulate all plots proportionally

from IPython.display import HTML, display # to display anim inline
from matplotlib.ticker import AutoLocator # for integer ticks/labels for entropy vs n plot
from scipy.stats import entropy # actually KLD method
np.seterr(divide='ignore', invalid='ignore')
#__________________________________________________________________________________________________________________________
class BoundWalk:
    #---------------------------------------------------------------------------------
    def __init__(self, total_steps: int,
                 step_bounds: tuple = (0.1, 1.0), # replaced step_size!
                 init_pos: float = 0,
                 boundaries: list = [0, 1], seed: int = None):
        self.X0 = init_pos
        self.N = total_steps
        self.step_bounds = step_bounds # to be used for uniform random sampling step_size!
        self.a = boundaries[0]
        self.b = boundaries[1]
        self.seed = seed

        self.decimals = max(0, -int(np.floor(np.log10(self.step_bounds[0])))) # derive rounding precision from lower bound of step_bounds!
        self.simulate() # simulate data for random walk, based on given parameters
    #---------------------------------------------------------------------------------
    @staticmethod
    def _reflect(pos, a, b):
        """
        Fold-back reflection: handles arbitrarily large overshoots
        by repeatedly folding until the position is inside [a, b].
        """
        while pos < a or pos > b:
            if pos < a:
                pos = 2*a - pos
            elif pos > b:
                pos = 2*b - pos
        return pos
    #---------------------------------------------------------------------------------
    def simulate(self): # GENERATOR for simulating random walk on given parameters
        rng = np.random.default_rng(self.seed)
        
        signs      = rng.choice([-1, 1], size=self.N)
        magnitudes = rng.uniform(self.step_bounds[0], self.step_bounds[1], size=self.N)
        displacements = signs * magnitudes
        
        positions = [self.X0]
        counts = defaultdict(int) # dict that keeps count of probs per outcome

        possible_outcomes = [np.array([self.X0])]   # particle is at X0 with certainty
        probabilities     = [np.array([1.0])]        # p(X0) = 1
        entropies         = [0.0]                    # H = 0, certain state
        norms             = [0.0]                    # KLD = 0, nothing has diverged yet

        # --- define the uniform reference vector ---
        domain = np.arange(self.a, self.b + 1e-8, self.step_bounds[0]) # now fixed by lower bound of step_bounds
        uniform_probs = np.ones(len(domain)) / len(domain) # needed for computing KLD
        ###########################################################
        def pad_to_domain(outcomes, probs):
            aligned = np.zeros(len(domain))
            for outcome, p in zip(outcomes, probs):
                idx = round((outcome - self.a) / self.step_bounds[0])
                if 0 <= idx < len(domain):
                    aligned[idx] = p
            total = aligned.sum()       # defensive renormalization
            if total > 0:               # guard against all-zero vector
                aligned /= total
            return aligned
        ###########################################################
        realized_displacements = [0] # to record displacements after corrections within reflective bounds [0,1]
        for step in displacements:
            raw_next = positions[-1] + step
            next_pos = self._reflect(raw_next, self.a, self.b)   # use the static method
            next_pos = round(next_pos, self.decimals)
            step     = round(next_pos - positions[-1], self.decimals)   # realized displacement, derived from actual positions
            positions.append(next_pos) 
            realized_displacements.append(step) # compute actual displacement after refleciton, then append

            counts[next_pos] += 1
            total = sum(counts.values())
            outcomes = np.array(list(counts.keys()))
            probs = np.array([counts[o] / total for o in outcomes])
            S = -(probs * np.log(probs)).sum() # after some point in development, dividing by 0 warning
            possible_outcomes.append(outcomes)
            probabilities.append(probs)
            entropies.append(S)

            # --- KLD: empirical vs uniform ---
            empirical_probs = pad_to_domain(outcomes, probs)
            norm_val = entropy(empirical_probs, uniform_probs) # compute KLD
            norms.append(norm_val)

        self.data = pd.DataFrame({ # df of generated data
            "n ≤ N": np.arange(self.N+1),
            "nth Displacement": realized_displacements,
            "nth Position": positions,
            "nth Possible Outcomes": possible_outcomes,
            "nth Probabilities": probabilities,
            "nth Entropy": entropies,
            "nth KLD": norms
        })
    #---------------------------------------------------------------------------------
    @staticmethod
    def _readable_tick_step(bin_width, domain_width, max_ticks=10):
        """
        Find the smallest round number >= bin_width that produces
        at most max_ticks labels across the domain.
        """
        raw       = domain_width / max_ticks
        magnitude = 10 ** math.floor(math.log10(raw))
        for multiplier in [1, 2, 2.5, 5, 10]:
            candidate = magnitude * multiplier
            if candidate >= bin_width:
                return candidate
        return magnitude * 10
    #---------------------------------------------------------------------------------
    @staticmethod
    def _readable_yticks(p_max, n_ticks=5):
        """
        Generate clean ytick values from 0 up to p_max,
        scaling dynamically to the current maximum empirical probability per frame.
        """
        if p_max <= 0:
            return np.array([0.0])
        raw       = p_max / n_ticks
        magnitude = 10 ** math.floor(math.log10(raw))
        for multiplier in [1, 2, 2.5, 5, 10]:
            step = magnitude * multiplier
            if step * n_ticks >= p_max:
                ticks = np.arange(0, p_max + step, step)
                ticks = ticks[ticks <= p_max * 1.05]
                return ticks                              # just return clean round ticks
        return np.linspace(0, p_max, n_ticks + 1)
    #---------------------------------------------------------------------------------
    def visualize(self): # WRAPPER for visualize (Position(N), Position Prob Hist, Position vs N) & Entropy vs N & KLD vs N
        # Runtime Configuration (RC) Settings
        plt.rcParams["animation.html"] = "jshtml" # adjust (RC) "animation.html" to render animation as interactive HTML widget inline

        fig = plt.figure(figsize=(14,9))
        gs = gridspec.GridSpec(4, 2, height_ratios=[1,1,1,1])
        #------------------------------------------------------
        #----------------------- ax_anim ---------------------- !!! 1st row, 1st col: 1D Position(N) !!!
        ax_anim = fig.add_subplot(gs[0,0]) # 1d bound rand walk anim
        position, = ax_anim.plot([], [], 'o', markersize=6, color='C0') # initialize scatter marker for anim
        position_text = ax_anim.text(0.01, 0.95, "", transform=ax_anim.transAxes, # dynamically updating text for position w.r.t. N
                                     fontsize=10, va="top", ha="left")
        ax_anim.axhline(0, color='black', linewidth=0.7, alpha=0.3) # reference line for y=0

        ax_anim.set_title(rf"$\mathcal{{BW}}[N={{{self.N}}}, |\Delta X| \sim \mathcal{{U}}[{{{self.step_bounds[0]}}}, {{{self.step_bounds[1]}}}]; X_0 \equiv {{{self.X0}}}]: \ x_{{_{{n}}}} \in \mathbb{{R}}_{{_{{{[self.a, self.b]}}}}}$")
        ax_anim.get_yaxis().set_visible(False) # don't need to see yaxis ticks/labels
        ax_anim.set_xlabel(rf"$X_{{_{{n}}}} = x_{{_{{n}}}} \in \mathbb{{R}}_{{_{{{[self.a, self.b]}}}}}$")
        ax_anim.set_ylim(-0.05, 0.1) # limit yaxis dimensions
        ax_anim.set_xlim(self.a, self.b) # walk will be confined within (a,b)
        #------------------------------------------------------
        #----------------------- ax_hist ---------------------- !!! 1st row, 2nd col: Probability Histogram !!!
        bin_width = self.step_bounds[0]   # local alias — bin width is the lower bound of step_bounds
        ax_hist = fig.add_subplot(gs[0,1]) # prob hist of positions
        centers = np.arange(self.a, self.b + 1e-8, bin_width) # [0, 1) partitions by 0.1 (default)
        edges = np.append(centers - bin_width/2, centers[-1] + bin_width/2) # each center has edges +/- 0.1 (default), be sure to include edge 1+0.05

        bars = ax_hist.bar(centers, np.zeros_like(centers),
                           width=bin_width, align='center',color="C0", edgecolor="black", alpha=0.7)
        ax_hist.axhline(1/len(centers), color="black", linestyle="--", linewidth=0.8,
                        label=f"U({self.a},{self.b})")

        ax_hist.set_title(r"$\mathbb{P}(X_{{_{{n}}}} = x_{{_{{n}}}})$ Histogram")
        ax_hist.set_ylabel(r"$\mathbb{P} \in \mathbb{{R}}_{{_{[0, 1]}}}$ ")
        ax_hist.set_xlabel(rf"$X_{{_{{n}}}} = x_{{_{{n}}}} \in \mathbb{{R}}_{{_{{{[self.a, self.b]}}}}}$")
        ax_hist.set_xlim(self.a - bin_width/2, self.b + bin_width/2)

        tick_step    = self._readable_tick_step(bin_width, self.b - self.a)
        tick_centers = np.arange(self.a, self.b + 1e-8, tick_step)
        ax_hist.set_xticks(tick_centers)
        ax_hist.set_xticklabels([f"{c:.{self.decimals}f}" for c in tick_centers])

        ax_hist.legend(loc='best') # enables label for U(a,b) PDF
        #------------------------------------------------------
        #------------------------------------------------------
        #----------------------- ax_plot ---------------------- !!! 2nd row: Position vs N plot !!!
        ax_plot = fig.add_subplot(gs[1, :]) # plot of positions vs n
        line_plot, = ax_plot.plot([], [], color="C0", alpha=0.7)
        marker_plot, = ax_plot.plot([], [], ".", color="C0")
        line_text = ax_plot.text(0.01, 0.95, "", transform=ax_plot.transAxes,
                                 fontsize=10, va="top", ha="left")

        ax_plot.set_title(r"$X_{{_{{n}}}} = x_{{_{{n}}}}$ vs $n \to N$")
        ax_plot.set_ylabel(r"$X_{{_{{n}}}} = x_{{_{{n}}}}$")
        ax_plot.set_ylim(self.a, self.b)
        ax_plot.tick_params(labelbottom=False)   # hide x tick labels
        #------------------------------------------------------
        #----------------------- ax_entr ---------------------- !!! 3rd row: Entropy vs N Plot !!!
        ax_entr = fig.add_subplot(gs[2,:], sharex=ax_plot) # entropy vs n
        line_entr, = ax_entr.plot([], [], color="C0", alpha=0.7)
        marker_entr, = ax_entr.plot([], [], ".", color="C0")
        text_entr = ax_entr.text(0.01, 0.95, "", transform=ax_entr.transAxes,
                                 fontsize=10, va="top", ha="left")
        ax_entr.axhline(np.log(len(centers)), color='red', linewidth=0.7, linestyle='--', alpha=0.7)  # Boltzmann Entropy supremum
        ax_entr.axhline(0, color='black', linewidth=0.7, alpha=0.3) # reference for y=0

        ax_entr.set_title(r"$H[X_{{_{{n}}}}]$ vs $n \to N$")
        ax_entr.set_ylabel(r"$H[X_{{_{{n}}}}]$")
        ax_entr.tick_params(labelbottom=False)   # hide x tick labels
        #------------------------------------------------------
        #----------------------- ax_norm ---------------------- !!! 4th row: KLD vs N Plot !!!
        ax_norm = fig.add_subplot(gs[3,:], sharex=ax_plot) # KLD vs N
        line_kld, = ax_norm.plot([], [], color="C0", alpha=0.7)
        marker_kld, = ax_norm.plot([], [], ".", color="C0")
        text_kld = ax_norm.text(0.01, 0.95, "", transform=ax_norm.transAxes,
                                fontsize=10, va="top", ha="left")
        ax_norm.axhline(0, color='black', linewidth=0.7, alpha=0.3) # reference for y=0

        ax_norm.set_title(r"$D_{{KL}}(\mathbb{P}||\mathcal{U})$ vs $n \to N$")
        ax_norm.set_ylabel(r"$D_{{KL}}(\mathbb{P}||\mathcal{U})$")
        ax_norm.set_xlabel(r"$n \to N$")
        #========================================================================================
        #======================== animate HELPER ================================================ # CHANGE name to something better!
        def animate(frame): # i within [0, len(walk.data)]
            #----------------------- ax_anim ---------------------- !!! 1st row, 1st col: 1D Position(N) !!!
            current_pos = self.data["nth Position"].iloc[frame] # find nth Position given n=frame
            position.set_data([current_pos], [0]) # move position marker to (x=current_pos, y=0)
            position_text.set_text(f"$X_{{_{{{frame}}}}}$ = {current_pos}") # update position_text label with current_pos
            #------------------------------------------------------
            #----------------------- ax_hist ---------------------- !!! 1st row, 2nd col: Probability Histogram !!!
            if frame > 0:
                current_positions = self.data["nth Position"].iloc[1: frame+1]
                counts, _ = np.histogram(current_positions, bins=edges)
                probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
            
                for bar, height in zip(bars, probs):
                    bar.set_height(height)
            
                ax_hist.xaxis.set_major_locator(AutoLocator())
            
            else: # frame=0 — particle is at X0 with certainty
                x0_idx = round((self.X0 - self.a) / bin_width)
                for i, bar in enumerate(bars):
                    bar.set_height(1.0 if i == x0_idx else 0.0)

            # --- dynamic y-axis ---
            if frame > 0:
                p_max  = probs.max() if probs.max() > 0 else 1.0
                yticks = self._readable_yticks(p_max)
                ax_hist.set_ylim(0, yticks[-1] * 1.05)
                ax_hist.set_yticks(yticks)
                ax_hist.set_yticklabels([f"{y:.2f}" for y in yticks])
            else:
                ax_hist.set_ylim(0, 1.05)
                ax_hist.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax_hist.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])

            ax_hist.set_xlim(self.a, self.b)
            #------------------------------------------------------
            #------------------------------------------------------
            #----------------------- ax_plot ---------------------- !!! 2nd row: Position vs N plot !!!
            x_vals = self.data['n ≤ N'].iloc[:frame+1]
            plot_yvals = self.data["nth Position"].iloc[:frame+1]

            line_plot.set_data(x_vals, plot_yvals)
            marker_plot.set_data([self.data['n ≤ N'].iloc[frame]], [self.data['nth Position'].iloc[frame]])
            line_text.set_text(fr"$X_{{_{{{frame}}}}} = $ {self.data['nth Position'].iloc[frame]}")


            ax_plot.relim()
            ax_plot.autoscale_view()
            ax_plot.xaxis.set_major_locator(AutoLocator()) # only integer x ticks!
            #------------------------------------------------------
            #----------------------- ax_entr ---------------------- !!! 3rd row: Entropy vs N Plot !!!
            xvals = self.data['n ≤ N'].iloc[1:frame+1]
            entr_y_vals = self.data['nth Entropy'].iloc[1:frame+1]

            line_entr.set_data(xvals, entr_y_vals)
            marker_entr.set_data([self.data['n ≤ N'].iloc[frame]], [self.data['nth Entropy'].iloc[frame]])

            text_entr.set_text(fr"$\text{{ln}}({{{len(np.arange(self.a, self.b + 1e-8, bin_width))}}}) = {{{(np.log(len(centers))).round(4)}}}$" + "\n" + fr"$H[X_{{_{{{frame}}}}}] = {self.data['nth Entropy'].iloc[frame]:.4f}$")

            ax_entr.relim()
            ax_entr.autoscale_view()
            ax_entr.xaxis.set_major_locator(AutoLocator()) # only integer x ticks!
            #------------------------------------------------------
            #----------------------- ax_norm ---------------------- !!! 4th row: KLD vs N Plot !!!
            kld_yvals = self.data['nth KLD'].iloc[1:frame+1]

            line_kld.set_data(xvals, kld_yvals)
            marker_kld.set_data([self.data['n ≤ N'].iloc[frame]], [self.data['nth KLD'].iloc[frame]])
            text_kld.set_text(fr"$D_{{KL}}(\mathbb{{P}}||\mathcal{{U}}) = {self.data['nth KLD'].iloc[frame]:.5f}$")

            ax_norm.relim()
            ax_norm.autoscale_view()
            ax_norm.xaxis.set_major_locator(AutoLocator()) # only integer x ticks!
        #========================================================================================
        plt.subplots_adjust(left=0.075, bottom=0.075, hspace=0.4)  # increase margins

        plt.close(fig) # ensure no static plots are displayed
        anim = animation.FuncAnimation(fig, animate,
                frames=len(self.data),
                interval=500, blit=False  # blitting doesn’t play well with clearing/replotting
            )

        # --- Display inline in notebook ---
        matplotlib.rcParams["animation.embed_limit"] = 50_000_000 # adjust (RC) for increasing animation file size to ~50 MB, may need to turn off for gif creation!
        display(HTML(anim.to_jshtml())) # for interactive HTML widget inline in notebook
    #---------------------------------------------------------------------------------
#__________________________________________________________________________________________________________________________