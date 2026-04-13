import numpy as np
from math import log10, floor


# For specifying step_size during object instantiation
def Uniform(a: float, b: float, bin_width: float) -> dict:
    return {'type': 'uniform', 'bounds': (a, b), 'bin_width': bin_width}

def Normal(mean: float, std_dev: float, bin_width: float) -> dict:
    return {'type': 'normal', 'params': (mean, std_dev), 'bin_width': bin_width}


# For determining rounding precision based on step_size
def find_rounding_precision(step_size: float | dict) -> int:
    if isinstance(step_size, float):
        return max(0, -int(floor(log10(step_size)))) 
    else:
        match step_size['type']:
            case 'uniform':
                return max(0, -int(floor(log10(step_size['bin_width'])))) 
            case 'normal':
                return max(0, -int(floor(log10(step_size['bin_width']))))


# For creating displacements based on given parameters
def generate_noise(seed: int, total_steps: int, step_size: float | dict) -> np.ndarray:
    RNG = np.random.default_rng(seed)

    if isinstance(step_size, float):
        return RNG.choice([-1, 1], size=total_steps) * step_size
    else:
        match step_size['type']:
            case 'uniform':
                magnitudes = RNG.uniform(step_size['bounds'][0], step_size['bounds'][1], size=total_steps)
                return RNG.choice([-1, 1], size=total_steps) * magnitudes
            case 'normal':
                return RNG.normal(step_size['params'][0], step_size['params'][1], size=total_steps)


# For determining bin width for histogram and domain definition
def get_bin_width(step_size: float | dict) -> float:
    if isinstance(step_size, float):
        return step_size
    else:
        match step_size['type']:
            case 'uniform':
                return step_size['bin_width']
            case 'normal':
                return step_size['bin_width']


# For reflecting raw next position back into the defined boundaries
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
    total = aligned.sum()
    return aligned / total if total > 0 else aligned
