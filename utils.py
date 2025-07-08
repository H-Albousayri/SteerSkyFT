import matplotlib
import os 
import random
import torch
import torch.nn as nn
import numpy as np
from cycler import cycler
from collections import deque
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi'] = 400

matplotlib.rcParams['axes.grid'] = True  # Enable grid by default
matplotlib.rcParams['grid.color'] = 'black'  # Set grid color
matplotlib.rcParams['grid.alpha'] = 1.0   # Set grid transparency
matplotlib.rcParams['grid.linestyle'] = (0, (6, 4))
matplotlib.rcParams['grid.linewidth'] = 0.5  # Thin grid lines

matplotlib.rcParams['xtick.minor.visible'] = True  # Show minor ticks on x-axis
matplotlib.rcParams['ytick.minor.visible'] = True  # Show minor ticks on y-axis

matplotlib.rcParams['xtick.direction'] = 'in'  # Tick direction (inwards)
matplotlib.rcParams['ytick.direction'] = 'in'

matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['#0000FF', 'r', 'g', 'k'])
matplotlib.rcParams['lines.linewidth'] = 2.0  # Standard line width

matplotlib.rcParams['font.size'] = 12  # Default font size
# matplotlib.rcParams['axes.titlesize'] = 12  # Title font size
matplotlib.rcParams['axes.labelsize'] = 15  # Label font size (x, y axis)
matplotlib.rcParams['xtick.labelsize'] = 13  # X-axis tick font size
matplotlib.rcParams['ytick.labelsize'] = 13  # Y-axis tick font size
matplotlib.rcParams['legend.fontsize'] = 10  # Legend font size
matplotlib.rcParams['figure.titlesize'] = 12  # Figure title font size



def get_exponential_noise(episode, total_episodes, initial_noise=0.45, final_noise=0.005):
    decay_rate = np.log(final_noise / initial_noise) / total_episodes
    noise = initial_noise * np.exp(decay_rate * episode)
    return max(noise, final_noise)


def set_deterministic(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[✓] Seeds set and deterministic behavior enforced with seed = {seed}")
