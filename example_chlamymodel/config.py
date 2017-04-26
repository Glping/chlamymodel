import numpy as np

frequency = 0.050
omega = 2 * np.pi * frequency
period = 2 * np.pi / omega
time_resolution = 100
dt = period / time_resolution
tau_A = 5.9
efficiency = 0.2
