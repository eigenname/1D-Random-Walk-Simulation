import importlib, bw_tools
importlib.reload(bw_tools) 
from bw_tools import ( # import all helper functions from bw_tools.py
    find_rounding_precision,
    generate_noise,
    pad_to_domain,
    reflect,
    get_bin_width,
)

import numpy as np # initially, to account for seed reproducibility
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd # for storing generated data in a structured format

from dataclasses import dataclass # for cleaner class definition
from collections import defaultdict # dict that keeps occurrence count for each position bin
from scipy.stats import entropy as KL_div # actually KLD method


@dataclass(kw_only=True) # use keyword-only arguments for clarity and to avoid confusion when instantiating the class with many parameters
class BoundWalk:
    total_steps: int # number of steps in the random walk
    step_size: tuple | float = None # size of step: can be randomly sampled (uniform or standard normal) or fixed (float)
    time_scale: float = 1.0 # default time scale for the walk
    boundaries: tuple = (0, 1) # default boundaries for the walk

    initial_position: float = 0 
    initial_momentum: float = 0 # mass is 1, for nats
    initial_energy: float = None # if set, overrides step_size

    seed: int = None # optional seed for reproducibility
    ms_between_frames: int = 500 # milliseconds between frames in animation, default is 500ms (0.5s)
    verbose: bool = True # show progress during simulation, default is True
    #__________________________________________________________________________________________________________________________
    def __post_init__(self): 
        self.__simulate__()
    #__________________________________________________________________________________________________________________________
    def __simulate__(self): # generates data for random walk, simulated based on given parameters
        if self.verbose:
            print(f"[BoundWalk] Initializing simulation...")

        Δt = self.time_scale
        init_position = self.initial_position
        init_momentum = self.initial_momentum # mass is 1 for 'nats'
        init_energy = self.initial_energy
        seed = self.seed
        N = self.total_steps
        left_bound, right_bound = self.boundaries

        if init_energy is not None and self.step_size is None:
            raw_step = np.sqrt(2 * init_energy) * Δt
            rounding_precision = find_rounding_precision(raw_step)
            step_size = round(raw_step, rounding_precision)
            # warn if rounding introduced significant error
            if abs(step_size - raw_step) / raw_step > 1e-6:
                print(f"[BoundWalk] Warning: step_size rounded from {raw_step:.10f} to {step_size} — consider adjusting initial_energy or time_scale.")
            self.step_size = step_size

        elif self.step_size is not None:
            step_size = self.step_size
            rounding_precision = find_rounding_precision(step_size)  # handles float or dict

        else:
            raise ValueError("Either step_size or initial_energy must be provided.")

        if self.verbose:
            print(f"[BoundWalk] Generating {N} noise samples...")
        self.white_noise = generate_noise(seed, N, step_size) # generate white noise displacements based on given parameters

        X = [init_position] 
        P = [init_momentum] 
        E = [0] # intentionally initialize as 0, separate from init_energy = self.initial_energy 
        counts_position = defaultdict(int) # dict that keeps count of probs per outcome
        Outcomes_positions = [np.array([init_position])] # particle is at X_0 with certainty
        Prob_positions     = [np.array([1.0])] # Prob(X_0) = 1 for n=0
        S_G = [0] # initialize Gibbs entropy @ 0 for n=0
        KLD = [0] # initialize D_KL(P||U)  @ 0 for n=0, nothing has diverged yet
 
        self.bin_width = get_bin_width(step_size) # determine bin width for histogram and domain definition, based on step_size argument
        self.domain = np.arange(left_bound, right_bound + 1e-8, self.bin_width) # define domain for histogram and D_KL(P||U)  computation
        uniform_probs = np.ones(len(self.domain)) / len(self.domain) # true distribution for computing D_KL(P||U) 

        if self.verbose:
            print(f"[BoundWalk] Running simulation: 0/{N} steps", end='', flush=True)
        ΔX = [0] # initialize displacement at n=0 as 0
        checkpoint_interval = max(1, N // 20)  # Progress tracking parameters, show progress every 5%
        for n, step in enumerate(self.white_noise, start=1):
            original_step = X[-1] + step # take a step
            next_position = round(reflect(original_step, left_bound, right_bound), rounding_precision) # reflect back into boundaries if step goes beyond, then round
            step = round(next_position - X[-1], rounding_precision) # compute actual displacement after reflection, then round
            
            X.append(next_position) # append actual next position 
            ΔX.append(step) # append actual step/displacement
            P.append(round(ΔX[-1] / Δt, max(rounding_precision+6, 10))) # compute and append next momentum
            E.append(round(P[-1]**2 / 2, max(rounding_precision+6, 10))) 

            counts_position[next_position] += 1 # update count for actual next position's bin
            total_positions = sum(counts_position.values()) # total count of all position outcomes so far, for normalizing probabilities
            unique_positions = np.array(list(counts_position.keys())) # unique position outcomes so far
            probabilities = np.array([counts_position[position] / total_positions for position in unique_positions]) # probabilities for each unique position outcome so far
            Outcomes_positions.append(unique_positions) # 
            Prob_positions.append(probabilities)
            S_G.append(-(probabilities * np.log(probabilities)).sum()) # compute Gibbs entropy using the probabilities of the outcomes at this step, then append to list

            empirical_probs = pad_to_domain(unique_positions, probabilities, self.domain, left_bound, self.bin_width) # align the outcomes and probabilities from each step with the defined domain, to get empirical distribution in the same support as uniform distribution for KLD computation
            KLD.append(KL_div(empirical_probs, uniform_probs)) # compute then append KLD for this step
            # Update progress inline
            if self.verbose and (n % checkpoint_interval == 0 or n == N):
                print(f"\r[BoundWalk] Running simulation: {n}/{N} steps ({100*n//N}%)", end='', flush=True)

        if self.verbose:
            print()  # New line after progress complete
            print(f"[BoundWalk] Building DataFrame...")
        self.data = pd.DataFrame({ # tabular data of simulation
            "t": np.arange(N+1) * Δt,
            "n ≤ N": np.arange(N+1),
            "nth Position": X,
            "nth Displacement": ΔX,
            "nth Momentum": P,
            "nth Energy": E,
            "nth Possible Positions": Outcomes_positions,
            "nth Position Probabilities": Prob_positions,
            "nth Entropy": S_G,
            "nth KLD": KLD
        })
        if self.verbose:
            print(f"[BoundWalk] Simulation complete. ({N} steps, {len(self.domain)} bins)\n")
    #__________________________________________________________________________________________________________________________
    def __visualize__(self, observable='position', verbose=None):
        """
        Create animated visualization of specified observable
        
        Parameters
        ----------
        observable : str, default 'position'
            Which observable to visualize:
            - 'position': X(t) trajectory + histogram + entropy + KLD (default)
            - 'momentum': P(t) trajectory + distribution + phase space
            - 'energy': E(t) conservation + distribution + phase space
        verbose : bool, optional
            Override self.verbose for this visualization
        
        Examples
        --------
        >>> walk = BoundWalk(total_steps=1000, step_size=0.1)
        >>> walk.visualize('position')  # Default thermalization view
        >>> walk.visualize('energy')    # Energy conservation view
        >>> walk.visualize('momentum')  # Momentum/phase space view
        """
        import bw_viz
        importlib.reload(bw_viz)
        
        if observable == 'position':
            bw_viz.visualize_position(self, verbose=verbose)
        elif observable == 'momentum':
            bw_viz.visualize_momentum(self, verbose=verbose)
        elif observable == 'energy':
            bw_viz.visualize_energy(self, verbose=verbose)
        else:
            raise ValueError(f"Unknown observable '{observable}'. Choose from: 'position', 'momentum', 'energy'")
    
    # Legacy alias for backward compatibility
    # def __visualize__(self, observable='position', verbose=None):
    #     """Deprecated: use .visualize() instead"""
    #     return self.visualize(observable=observable, verbose=verbose)
    #__________________________________________________________________________________________________________________________
