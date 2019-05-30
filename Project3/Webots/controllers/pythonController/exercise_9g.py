"""Exercise 9g"""

import numpy as np
import matplotlib.pyplot as plt 
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9g(world, timestep, reset):
    """Exercise 9g"""
    
    parameters = SimulationParameters(
            simulation_duration=60, 
            amplitude_gradient = [1,1], 
            phase_lag=2*np.pi/10, 
            turn=[1, 'Right'],
#            offset = 2*np.pi/10, #optimal phase lag
            amp_factor = 1.0, 
            drive_mlr = 1.5
            ) 
    run_simulation(
                world,
                parameters,
                timestep,
                int(1000*parameters.simulation_duration/timestep),
                logs="logs/simulation9g.npz"
            )
    

