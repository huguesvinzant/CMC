"""Exercise 9f"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9f(world, timestep, reset):
    """Exercise 9f"""
    parameters = [SimulationParameters(
            simulation_duration=10, 
            drive=1.5, 
            amplitude_gradient = [0.2,1], 
            phase_lag=2*np.pi/10, 
            turn=[1, 'Right'],
            offset = phi
            # ...
        ) for phi in [np.pi, np.pi/4, np.pi/2]
    ]
    
    for simulation_i, parameters in enumerate(parameters):
            reset.reset()
            run_simulation(
                world,
                parameters,
                timestep,
                int(1000*parameters.simulation_duration/timestep),
                logs="./logs/simulation9f_{}.npz".format(simulation_i)
            )

