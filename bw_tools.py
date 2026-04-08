import numpy as np
from math import log10, floor, sqrt


# For specifying step_size during object instantiation
def Uniform(a: float, b: float, bin_width: float) -> dict: # function to create a dictionary representing a uniform distribution with given bounds (a, b)
    return {'type': 'uniform', 'bounds': (a, b), 'bin_width': bin_width}

def Normal(mean: float, std_dev: float, bin_width: float) -> dict: # function to create a dictionary representing a normal distribution with given mean (mu) and standard deviation (sigma)
    return {'type': 'normal', 'params': (mean, std_dev), 'bin_width': bin_width}


# For determining rounding precision based on step_size, whether to be fixed or randomly sampled
def find_rounding_precision(step_size: (float | int) | dict) -> int:
    if isinstance (step_size, (float, int)): # if step_size is float or int, is fixed
        return max(0, -int(floor(log10(step_size))))
    
    else: # if step_size is dict, step_size is to be randomly sampled
        match step_size['type']:
            case 'uniform': # from uniform distribution, derive rounding precision from lower bound of its support
                return max(0, -int(floor(log10(step_size['bin_width'])))) 
            case 'normal': # from a standard normal distribution, derive rounding precision from its standard deviation
                return max(0, -int(floor(log10(step_size['bin_width'])))) 


# For creating displacements based on given parameters, whether step_size is fixed or randomly sampled
def create_displacements(seed: int, total_steps: int, step_size: (float | int) | dict, time_scale: float) -> np.ndarray:
    RNG = np.random.default_rng(seed) # initialize seed, if given

    if isinstance (step_size, (float, int)): # if step_size is float or int, is fixed
        return RNG.choice([-1, 1], size=total_steps) * step_size * sqrt(time_scale) # random choice of left or right, multiplied by fixed step_size
    
    else: # if step_size is dict, step_size is to be randomly sampled
        match step_size['type']:
            case 'uniform': # generate random magnitudes from uniform distribution
                magnitudes = RNG.uniform(step_size['bounds'][0], step_size['bounds'][1], size=total_steps) # random magnitudes from uniform distribution
                return RNG.choice([-1, 1], size=total_steps) * magnitudes * sqrt(time_scale)
            
            case 'normal': # generate random magnitudes from normal distribution
                return RNG.normal(step_size['params'][0], step_size['params'][1], size=total_steps) * sqrt(time_scale) # random magnitudes from normal distribution

# For determining bin width for histogram and domain definition, based on step_size parameters, whether to be fixed or randomly sampled
def get_bin_width(step_size: (float | int) | dict) -> float:
    if isinstance (step_size, (float, int)): # if step_size is float or int, is fixed
        return step_size
        
    else: # if step_size is dict, step_size is to be randomly sampled
        match step_size['type']:
            case 'uniform': # from uniform distribution, derive bin width from lower bound of its support
                return step_size['bin_width']
            case 'normal': # from a standard normal distribution, derive bin width from its standard deviation
                return step_size['bin_width'] # if step_size is a distribution, derive bin width from standard deviation!
                
# For reflecting raw next position back into the defined boundaries, accounts for large step_sizes (beyond right_bound - left_bound)
def reflect(pos: float, left_bound: float, right_bound: float) -> float:
    while pos < left_bound or pos > right_bound:
        if pos < left_bound:
            pos = 2 * left_bound - pos
        elif pos > right_bound:
            pos = 2 * right_bound - pos

    return pos

# For aligning the outcomes and probabilities from each step with the defined domain
def pad_to_domain(outcomes: np.ndarray, probs: np.ndarray, domain: np.ndarray, left_bound: float, delta: float) -> np.ndarray:
    aligned = np.zeros(len(domain))
    for outcome, p in zip(outcomes, probs):
        idx = round((outcome - left_bound) / delta)
        if 0 <= idx < len(domain):
            aligned[idx] = p

    total = aligned.sum()  # defensive renormalization
    return aligned / total if total > 0 else aligned
