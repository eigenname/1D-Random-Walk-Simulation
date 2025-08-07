import math, numpy as np, matplotlib.pyplot as plt  # main modules
from matplotlib.ticker import MaxNLocator   # for integer tick formatting on plots
from matplotlib.gridspec import GridSpec    # for figure formatting for subplots
from scipy.stats import gaussian_kde    # KDE for generated distributions
from IPython.display import display, Math   # for printing/labeling in LaTeX
from scipy.optimize import curve_fit    # fitting logarithmic model to entropy        
#==============================================================================================================================================================================================================

class Walk: # Random walk method defined by steps taken, initial position, and random seed index
    #------------------------------------------------------------------------------------------------------------------------
    def __init__(self,steps, init_pos=None, seed=None, recep=None, bound=None): # verifies inputs, generates walk, and determines net displacement. recep is switch for initial_reception 
        if bound is not None:
            self.bound = round(bound, 2)
        else:
            self.bound = bound
        self.initial_reception(steps, init_pos, seed)
        self.generate_walk() 
        self.get_stats(recep)
    #------------------------------------------------------------------------------------------------------------------------
    def initial_reception(self, steps, init_pos, seed): # verification of initial data entry from user
        # default description for an individually generated random walk
        if steps < 1: # should be greater than 1, standard deviation is 0 by default for taking 1 step.
            raise ValueError("Cannot take a walk of < 1 steps.")
        self.n = steps # number of steps in a walk

        if init_pos is None: # if initial position is NOT given,
            init_pos = 0    # by default, initial position is 0
            self.x0 = init_pos
            print(f"No initial position was specified.") 
        else: # if initial position is given
            self.x0 = init_pos # initial_pos
        
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
            print(f"@seed{self.seed}.")
    #------------------------------------------------------------------------------------------------------------------------
    def generate_walk(self):    # creates [arr] of left/right steps and position [arr] for a walk
        self.left_right = np.random.choice([-1, 1], size=self.n, p=[0.5, 0.5]) # values for taking a left or right step, need to be able to access for constrained boundaries later
        self.positions_0 = np.array([self.x0]) # position [arr] including init pos
        if self.bound is not None:   # with boundary conditions
            for _ in range(self.n):
                if abs(self.positions_0[-1] + self.left_right[_]) > self.bound:
                    self.positions_0 = np.append(self.positions_0, self.positions_0[-1] - self.left_right[_])
                else:
                    self.positions_0 = np.append(self.positions_0, self.positions_0[-1] + self.left_right[_])
        else:   # without boundary conditions
            for _ in range(self.n):
                self.positions_0 = np.append(self.positions_0, self.positions_0[-1] + self.left_right[_])
    #------------------------------------------------------------------------------------------------------------------------
    def get_stats(self, recep=None):    # determines net displacement, unique outcome values and relative frequencies for a walk's positions
        self.net_displacement = self.positions_0[-1] - self.x0 # net displacement, important for gathering average net disaplacement w.r.t. n
        self.positions = np.delete(self.positions_0, 0) # needed for box & whisker and histogram
        self.uniq_outcomes, raw_freq = np.unique(self.positions, return_counts=True)
        self.rel_freq = raw_freq / len(self.positions)
        if recep == "off":   # no description if nots
            self.mean = round(np.mean(self.positions), 4)
            self.stdev = round(np.std(self.positions), 4)
        else:   # by default, provide
            self.mean = round(np.mean(self.positions), 4)
            self.stdev = round(np.std(self.positions), 4)
            display(Math(rf"\mathcal{{RW}}({{{self.n}}}, {{{self.x0}}}): \mu\pm\sigma \approx {{{self.mean}}} \pm {{{self.stdev}}}."))
        self.S = entropy(self.uniq_outcomes, self.rel_freq)
    #------------------------------------------------------------------------------------------------------------------------
    def plot_walk(self, desc=None):    # graph box and whisker plot, position histogram, and positions vs steps for a random walk
        fig = plt.figure(figsize=(15, 4))
        gs = GridSpec(2, 
                      2,
                      width_ratios=[1, 2],
                      height_ratios=[0.2, 0.8],
                      figure=fig)
        
        # define axes
        ax_bw = fig.add_subplot(gs[0, 0])  # 1) 1st row, 1st col: box-whisker plot
        ax_hist = fig.add_subplot(gs[1, 0], sharex=ax_bw)  # 2) 2nd row, 1st col: histogram
        ax_pos = fig.add_subplot(gs[0:2, 1])  # 3) 1st 2 rows, 2nd col: positions vs steps plot

        xmin, xmax = self.uniq_outcomes.min() - 0.5, self.uniq_outcomes.max() + 0.5 # x-limits for box & whisker and histogram
        # 1) Box-whisker plot
        ax_bw.boxplot(self.positions, vert=False)
        ax_bw.set_xlim(xmin, xmax)
        ax_bw.set_yticks([])
        # 2) Histogram of net displacements
        ax_hist.bar(self.uniq_outcomes, self.rel_freq, alpha=0.7, width=1.0, align="center", edgecolor="black")
        ax_hist.set_title("Position Histogram")
        ax_hist.set_ylabel("Relative Frequency"); ax_hist.set_xlabel("Position")
        ax_hist.set_xticks(self.uniq_outcomes)
        ax_hist.set_xlim(xmin, xmax)
        ax_hist.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
        # KDE 
        KDE = gaussian_kde(self.positions)
        domain_vals_KDE = np.linspace(self.uniq_outcomes.min()-3*np.std(self.uniq_outcomes), self.uniq_outcomes.max()+3*np.std(self.uniq_outcomes), 1000)
        ax_hist.plot(domain_vals_KDE, KDE(domain_vals_KDE), label="KDE", color="orange")
        ax_hist.legend(loc="best")


        # 3) Positions vs Step Count Plot
        ax_pos.plot(self.positions_0, linewidth=0.75, color="tab:blue", alpha=1)
        if desc != "off":
            if self.bound is not None:
                fig.suptitle(f"A 1D random walk starting at {self.x0} for {self.n} steps, within |$\pm$ {self.bound}|.", fontsize=14)
                ax_pos.set_ylim(-self.bound-0.1, self.bound+0.1)
                ax_pos.axhline(y=-self.bound, color="black", linewidth=0.75, linestyle='-', alpha=1)
                ax_pos.axhline(y=self.bound, color="black", linewidth=0.75, linestyle='-', alpha=1, label=fr"|$\pm${self.bound}|")
                ax_pos.legend(loc="best")
            else:
                fig.suptitle(f"A 1D random walk starting at {self.x0} for {self.n} steps.", fontsize=14)
            
        ax_pos.set_xlim(0, self.n)
        ax_pos.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # prune removes cluttered edge ticks
        ax_pos.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # same for y-axis
        ax_pos.axhline(y=0, color="black", linewidth=0.75, linestyle='-', alpha=0.3)
        ax_pos.axvline(x=0, color="black", linewidth=0.75, linestyle='-')
        ax_pos.set_ylabel("Position"); ax_pos.set_xlabel("Step Count")
        ax_pos.set_title("Positions vs Step Count")
        
        plt.tight_layout()
        plt.show()
    #------------------------------------------------------------------------------------------------------------------------
    def plot_entropy(self, desc=None):     # separate graph for entropy vs positions. By default, desc provide description
        fig, ax = plt.subplots(figsize=(15, 4))
        if desc != "off":
            fig.suptitle(f"Entropy for a 1D random walk starting at {self.x0} for {self.n} steps.", fontsize=14)
        
        ax.scatter(self.uniq_outcomes, self.S, color="purple", marker=".", alpha=0.7)
        ax.plot(self.uniq_outcomes, self.S, color="purple", linestyle="--", alpha=0.7)

        ax.set_xlim(self.uniq_outcomes.min()-0.1, self.uniq_outcomes.max()+0.1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
        ax.axhline(y=0, color="black", linewidth=0.75, linestyle='-', alpha=0.3)
        ax.axvline(x=0, color="black", linewidth=0.75, linestyle='-', alpha=0.3)
        ax.set_ylabel(r"$S := S[p(x)]$"); ax.set_xlabel("Positions")
        ax.set_title(r"$S$ vs Position")

        plt.tight_layout()
        plt.show()
    #------------------------------------------------------------------------------------------------------------------------

#==============================================================================================================================================================================================================
# def gauss_appox(dom_vals, steps):   # Gaussian approximate from using Stirling's formula
#     return np.array([1/math.sqrt(2*np.pi*steps) * math.e**(-(dom_vals[_]**2)/(2*steps)) for _ in range(len(dom_vals))])

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x-mu)**2 / (2*sigma**2))
#==============================================================================================================================================================================================================
def entropy(domain_vals, probs):    # provide net displacement outcomes & relative frequencies --> entropy domain and range
    S = np.array([])
    for _ in range(len(domain_vals)):
        if _ == 0:
            S = np.append(S, -probs[_] * np.log(probs[_]))
        else:
            S = np.append(S, -sum(probs[0:_+1] * np.log(probs[0:_+1])))

    return S
#==============================================================================================================================================================================================================

class Walk_Trial:  # trial of net displacement measurements for M random walks, all for n steps starting at x0
    #------------------------------------------------------------------------------------------------------------------------
    def __init__(self, num_Walks, num_steps, init_pos, recep=None, bound=None):
        self.M = num_Walks
        self.n = num_steps
        self.x0 = init_pos
            
        if recep == "off":
            if bound is not None:
                self.bound= round(bound, 2)
                self.trial = np.array([Walk(self.n, self.x0, recep="off", bound=bound) for _ in range(self.M)])  # recep is "off" b/c generating multiple random walks
            else:
                self.trial = np.array([Walk(self.n, self.x0, recep="off") for _ in range(self.M)])  # recep is "off" b/c generating multiple random walks
                self.bound = bound
        self.measurements(recep)
    #------------------------------------------------------------------------------------------------------------------------
    def measurements(self, range_meas=None):    # collecting net displacement per random walk
        self.net_disps = self.means = self.stdevs = np.array([])   # net displacements for M random walks of n steps each
        for _ in range(self.M):
            self.net_disps = np.append(self.net_disps, self.trial[_].net_displacement)
            self.means = np.append(self.means, np.mean(self.net_disps))
            self.stdevs = np.append(self.stdevs, np.std(self.net_disps))

        self.mean = round(np.mean(self.net_disps), 1)   # mean for M walks
        self.stdev = round(np.std(self.net_disps), 1)   # stdev for M walks

        self.uniq_outcomes, raw_freq = np.unique(self.net_disps, return_counts=True)
        self.rel_freq = raw_freq / len(self.net_disps)

        if range_meas is None:  # by default, provide trial description, display 1st and last random walk measurements
            display(Math(rf"For \ n = {self.n}, \ \exists \ \mathcal{{RW}}_{1}(n, x_0), ..., \mathcal{{RW}}_{{{self.M}}}(n, x_0) \ni \mu\pm\sigma \approx {self.mean} \pm {self.stdev}."))    

        self.S = entropy(self.uniq_outcomes, self.rel_freq)
    #------------------------------------------------------------------------------------------------------------------------
    def plot_trial(self):   # plot box and whisker, net displacement histogram, all walk positions vs steps. By default, desc provides description
        fig = plt.figure(figsize=(18, 8))
        gs = GridSpec(3, 
                      2,
                      width_ratios=[1, 2],
                      height_ratios=[0.2, 0.8, 1.2],
                      figure=fig)
        if self.bound is None:
            fig.suptitle(f"For {self.n} steps, {self.M} random walks starting at {self.x0} were generated.", fontsize=14)
        else:
            fig.suptitle(fr"For {self.n} steps, {self.M} random walks starting at {self.x0} and confined within |$\pm${self.bound}| were generated.", fontsize=14)
        
        # define axes
        ax_bw = fig.add_subplot(gs[0, 0])  # 1) 1st row, 1st col: box-whisker plot
        ax_hist = fig.add_subplot(gs[1, 0], sharex=ax_bw)  # 2) 2nd row, 1st col: histogram
        ax_pos = fig.add_subplot(gs[0:2, 1])  # 3) 1st 2 rows, 2nd col: all walk positions vs steps plot
        ax_disps = fig.add_subplot(gs[2, :])  # 4) 3rd row, all cols: net displacements vs steps

        xmin, xmax = self.uniq_outcomes.min()-0.5, self.uniq_outcomes.max()+0.5 # x-limits for box & whisker and histogram
        # 1) Box-whisker plot
        ax_bw.boxplot(self.net_disps, vert=False)   # box & whisker plot
        ax_bw.set_xlim(xmin, xmax)
        ax_bw.set_yticks([])
        # 2) Histogram of net displacements
        ax_hist.bar(self.uniq_outcomes, self.rel_freq, alpha=0.7, width=1.0, align="center", edgecolor="black") # bar graph of unique net displacement outcomes
        ax_hist.set_title("Net Displacement Histogram")
        ax_hist.set_ylabel("Relative Frequency"); ax_hist.set_xlabel("Net Displacement")
        ax_hist.set_xticks(self.uniq_outcomes)
        ax_hist.set_xlim(xmin, xmax)
        ax_hist.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
        # KDE and fitting Gaussian approximate
        self.domain_gauss = np.linspace(self.uniq_outcomes.min()-3*self.stdev, self.uniq_outcomes.max()+3*self.stdev, 1000)
        ax_hist.plot(self.domain_gauss, self.fit_gauss(), # gaussian
                     linewidth=2, color="red", 
                     label=rf"$\mathcal{{N}}({self.params[1]:.2f}, \ {self.params[2]**2:.2f})$") 
        ax_hist.legend(loc="best")

        # 3) Positions vs Step Count Plot
        for _ in range(self.M):
            ax_pos.plot(self.trial[_].positions_0, linewidth=0.75, color="tab:blue", alpha=0.7)

        if self.bound is not None:
            ax_pos.set_ylim(-self.bound-0.1, self.bound+0.1)
            ax_pos.axhline(y=-self.bound, color="black", linewidth=0.75, linestyle='-', alpha=1)
            ax_pos.axhline(y=self.bound, color="black", linewidth=0.75, linestyle='-', alpha=1, label=fr"|$\pm${self.bound}|")
            ax_pos.legend(loc="best")
        ax_pos.set_xlim(0, self.n)
        ax_pos.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # prune removes cluttered edge ticks
        ax_pos.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # same for y-axis
        ax_pos.axhline(y=0, color="black", linewidth=0.75, linestyle='-', alpha=0.3)
        ax_pos.axvline(x=0, color="black", linewidth=0.75, linestyle='-')
        ax_pos.set_ylabel("Position"); ax_pos.set_xlabel("Step Count")
        ax_pos.set_title(f"{self.M} Walk Positions vs Step Count")

        # 4) Net Displacements vs Walk Count Plot
        ax_disps.scatter(np.array([_ for _ in range(1, self.M+1)]), self.net_disps, marker=".") # net displacements vs n
        ax_disps.plot(np.array([_ for _ in range(1, self.M+1)]), self.means, color="orange")    # mean of net displacements
        ax_disps.fill_between(np.array([_ for _ in range(1, self.M+1)]), self.means+self.stdevs, self.means-self.stdevs, # zoning between +/- stdev
                              color="orange", alpha=0.3, 
                              label=rf"$\mu \pm \sigma \rightarrow {self.mean:.2f} \pm {self.stdev}$")
        
        if self.bound is not None:
            ax_disps.set_ylim(-self.bound-0.1, self.bound+0.1)
            ax_disps.axhline(y=-self.bound, color="black", linewidth=0.75, linestyle=':', alpha=1)
            ax_disps.axhline(y=self.bound, color="black", linewidth=0.75, linestyle=':', alpha=1, label=fr"|$\pm${self.bound}|")
            ax_disps.legend(loc="best")
        
        ax_disps.plot(np.array([_ for _ in range(1, self.M+1)]), np.array([math.sqrt(self.n) for _ in range(1, self.M+1)]), # 0 + sqrt(n)
                      color="red", linewidth=0.75, linestyle="--")
        ax_disps.plot(np.array([_ for _ in range(1, self.M+1)]), np.array([-math.sqrt(self.n) for _ in range(1, self.M+1)]), # 0 - sqrt(n)
                      color="red", linewidth=0.75, linestyle="--", 
                      label=rf"$\pm \sqrt{{n}} \approx \pm {math.sqrt(self.n):.2f}$")

        ax_disps.set_xlim(0.9, self.M+0.1)
        if self.bound is not None:
            ax_disps.set_ylim(-self.bound-1, self.bound+1)
        ax_disps.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # prune removes cluttered edge ticks
        ax_disps.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # same for y-axis
        ax_disps.set_ylabel("Net Displacement"); ax_disps.set_xlabel("Walk Count")
        ax_disps.set_title("Net Displacement vs Walk Count")
        ax_disps.legend(loc="best")

    
        plt.tight_layout()
        plt.show()
    #------------------------------------------------------------------------------------------------------------------------
    def plot_entropy(self, desc=None):     # separate graph for entropy vs net displacements. By default, desc provide description
        fig, ax = plt.subplots(figsize=(15, 4))
        if desc is None:
            fig.suptitle(f"Entropy for {self.M} random walks starting at {self.x0} for {self.n} steps each.", fontsize=14)
        
        ax.scatter(self.uniq_outcomes, self.S, color="purple", marker=".", alpha=0.7)
        ax.plot(self.uniq_outcomes, self.S, color="purple", linestyle="--", alpha=0.7)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
        ax.axhline(y=0, color="black", linewidth=0.75, linestyle='-', alpha=0.3)
        ax.axvline(x=0, color="black", linewidth=0.75, linestyle='-', alpha=0.3)
        ax.set_ylabel(r"$S := S[p(x_f)]$"); ax.set_xlabel("Net Displacements")
        ax.set_title(r"$S$ vs Net Displacements")

        plt.tight_layout()
        plt.show()
    #------------------------------------------------------------------------------------------------------------------------
    def fit_gauss(self):
        self.params, _ = curve_fit(gaussian, self.uniq_outcomes, self.rel_freq)
        return gaussian(self.domain_gauss, *self.params)
    #------------------------------------------------------------------------------------------------------------------------

#==============================================================================================================================================================================================================

class List_Trials(): # list of trials of net displacement measurements, as n becomes large
    #------------------------------------------------------------------------------------------------------------------------
    def __init__(self, num_trials, walks_per_trial, steps_per_trial, init_pos, recep=None, bound=None):
        self.t = num_trials
        self.M = walks_per_trial
        self.n = steps_per_trial
        self.x0 = init_pos
        if bound is not None:
            self.bound = round(bound, 2)
        else:
            self.bound = bound

        if self.bound is not None:
            self.trials = np.array([Walk_Trial(walks_per_trial[_], steps_per_trial[_], init_pos, recep="off", bound=bound) for _ in range(len(steps_per_trial))])    # list of trials
        else:
            self.trials = np.array([Walk_Trial(walks_per_trial[_], steps_per_trial[_], init_pos, recep="off") for _ in range(len(steps_per_trial))])
        
        self.stdevs = np.array([self.trials[_].stdev for _ in range(num_trials)]) # standard deviation for each trial of measurements
        self.S = np.array([self.trials[_].S.max() for _ in range(num_trials)])  # max entropy for the t-th trial of M walks for n steps

        self.log_dom_min = 1
        self.ext_domain = self.n.max()+50    # for extending domain of log predictions
        self.nth_ord = 16   # at 16, predictions tend to grow logarithmically w/o bound. if > 16, model starts underperforming
    #------------------------------------------------------------------------------------------------------------------------
    def plot_stdevs(self):  # plot standard deviation vs n large
        if self.bound is not None:
            fig, ax = plt.subplots(figsize=(15, 6))
            fig.suptitle(fr"For $n \in  [{self.n.min()}, {self.n.max()}] $, {self.t} trials were collected where each trial had {self.M[0]} walks confined within $|\pm {self.bound}|$ generated.", fontsize=16)
            
            ax.scatter(self.n, self.stdevs, color="tab:orange", edgecolors='black', linewidths=0.5, alpha=0.5)
            ax.plot(np.array([_ for _ in range(np.max(self.n))]), np.sqrt([_ for _ in range(np.max(self.n))]), 
                    color="black", linestyle="-", linewidth=1, alpha=1, label=r"$\sqrt{{n}}$")

            ax.plot(np.linspace(self.log_dom_min, self.ext_domain, self.n.max()+1), self.fit_log(self.stdevs), # fitted log model
                color="#FF7300", linestyle="--", linewidth=1, alpha=1, label="log") 
            ax.plot(np.linspace(self.log_dom_min, self.ext_domain, self.n.max()+1), self.fit_O2_log(self.stdevs), # fitted O^2 log model
                color="red", linestyle="--", linewidth=1, alpha=1, label=r"$O^2$(log)")   
            ax.plot(np.linspace(self.log_dom_min, self.ext_domain, self.n.max()+1), self.fit_On_log(self.nth_ord, self.stdevs), # fitted O^n log model
                color="green",  linewidth=1, alpha=1, label=rf"$O^{{{self.nth_ord}}}$(log)")  

            ax.set_ylim(0, self.stdevs.max()+0.5)
            ax.set_xlim(-0.5, np.max(self.n)+5)
            ax.axhline(y=0, color="black", linewidth=0.75, linestyle='-', alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both') )
            ax.set_title(r"$\sigma$ vs $n$")
            ax.set_ylabel(r"$\sigma$"); ax.set_xlabel(r"$n \rightarrow \infty$")
            ax.legend(loc="best")
    
            plt.tight_layout()
            plt.show()

        else:
            fig, ax = plt.subplots(figsize=(15, 6))
            fig.suptitle(fr"For $n \in  [{self.n.min()}, {self.n.max()}] $, {self.t} trials were collected where each trial had {self.M[0]} walks generated.", fontsize=16)

            ax.scatter(self.n, self.stdevs, color="tab:orange", edgecolors='black', linewidths=0.5, alpha=0.5)
            ax.plot(np.array([_ for _ in range(np.max(self.n))]), np.sqrt([_ for _ in range(np.max(self.n))]), color="black", linestyle="-", linewidth=0.75, label=r"$\sqrt{{n}}$")

            ax.set_xlim(-0.5, np.max(self.n)+5)
            ax.axhline(y=0, color="black", linewidth=0.75, linestyle='-', alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both') )
            ax.set_title(r"$\sigma$ vs $n$")
            ax.set_ylabel(r"$\sigma$"); ax.set_xlabel(r"$n \rightarrow \infty$")
            ax.legend(loc="best")

            plt.tight_layout()
            plt.show()
    #------------------------------------------------------------------------------------------------------------------------
    def plot_entropy(self): # plot max entropy of each trial of walks, of same steps taken and intial position, vs n becoming large
        fig, ax = plt.subplots(figsize=(15,4))
        if self.bound is not None:
            fig.suptitle(fr"For $n \in  [{self.n.min()}, {self.n.max()}] $, {self.t} trials were collected where each trial had {self.M[0]} walks confined within $|\pm{self.bound}|$ generated.", fontsize=14)
        else:
            fig.suptitle(fr"For $n \in  [{self.n.min()}, {self.n.max()}] $, {self.t} trials were collected where each trial had {self.M[0]} walks generated.", fontsize=14)

        ax.scatter(self.n, self.S, color="purple", marker=".", edgecolor="black", alpha=0.7)

        ax.plot(np.linspace(self.log_dom_min, self.ext_domain, self.n.max()+1), self.fit_log(self.S), # fitted log model
                color="#4530FF", linestyle="--", linewidth=1, alpha=1, label="log") 
        ax.plot(np.linspace(self.log_dom_min, self.ext_domain, self.n.max()+1), self.fit_O2_log(self.S), # fitted O^2 log model
                color="red", linestyle="--", linewidth=1, alpha=1, label=r"$O^2$(log)")   
        ax.plot(np.linspace(self.log_dom_min, self.ext_domain, self.n.max()+1), self.fit_On_log(self.nth_ord, self.S), # fitted O^n log model
                color="green",  linewidth=1, alpha=1, label=rf"$O^{{{self.nth_ord}}}$(log)")   

        ax.axhline(y=0, color="black", linewidth=0.75, linestyle='-', alpha=0.3)
        ax.axvline(x=0, color="black", linewidth=0.75, linestyle='-', alpha=0.3)
        ax.set_ylim(self.S.min()-0.2, self.S.max()+0.1)
        ax.set_xlim(-0.5, self.n.max()+10)

        ax.set_title(r"max$(S)$ vs $n$")
        ax.set_ylabel(r"max$(S[p(x_f)])$"); ax.set_xlabel(r"$n \rightarrow \infty$")
        ax.legend(loc="best")

        plt.tight_layout()
        plt.show()
    #------------------------------------------------------------------------------------------------------------------------
    def fit_log(self, range):  # fitting logarithmic model for max entropy w.r.t. n, and stdevs w.r.t n with boundaries
        params, _ = curve_fit(log_model, self.n, range)
        return log_model(np.linspace(self.log_dom_min, self.ext_domain, self.n.max()+1), *params)
    #------------------------------------------------------------------------------------------------------------------------
    def fit_O2_log(self, range):   # fit 2nd-Order log model for comparison
        params, _ = curve_fit(O2_log_model, self.n, range)
        return O2_log_model(np.linspace(self.log_dom_min, self.ext_domain, self.n.max()+1), *params)
    #------------------------------------------------------------------------------------------------------------------------
    def fit_On_log(self, nth_ord, range):   # fit nth-Order log model for best fit
        params, _ = curve_fit(On_log_model, self.n, range, p0=[0.1]*(nth_ord+1))
        return On_log_model(np.linspace(self.log_dom_min, self.ext_domain, self.n.max()+1), *params)
    #------------------------------------------------------------------------------------------------------------------------

#==============================================================================================================================================================================================================
def even_perfect_squares(num):    # option of generating even perfect squares for n, to evaluate for standard deviation 
    result = []
    i = 1
    while len(result) < num:
        square = i * i
        if square % 2 == 0:
            result.append(square)
        i += 1
    return np.array(result)
#==============================================================================================================================================================================================================
def even_nums(num):     # option for generating even integers for n, to evaluate entropy 
    return np.array([2*_ for _ in range(1, num+1)])
#==============================================================================================================================================================================================================
def log_model(x, scale, intercept):
    return scale*np.log(x) + intercept
#==============================================================================================================================================================================================================
def O2_log_model(x, scale1, scale2, intercept):
    return scale1*np.log(x) + scale2*(np.log(x))**2 + intercept
#==============================================================================================================================================================================================================
def On_log_model(x, *coeffs):
    return sum(scale * np.log(x)**ord for ord, scale in enumerate(coeffs))
#==============================================================================================================================================================================================================