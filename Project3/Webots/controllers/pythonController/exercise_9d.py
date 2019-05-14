"""Exercise 9d"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9d1(world, timestep, reset):
    """Exercise 9d1"""
    parameters = SimulationParameters(
            simulation_duration=10, 
            drive=4.5, 
            amplitude_gradient = [0.2,1], 
            phase_lag=2*np.pi/10, 
            turn=[0.2, 'Right'],
            # ...
        )
    run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/simulation9d1.npz")


def exercise_9d2(world, timestep, reset):
    """Exercise 9d2"""
    pass

