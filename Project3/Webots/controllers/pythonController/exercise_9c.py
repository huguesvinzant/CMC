"""Exercise 9c"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9c(world, timestep, reset):
    """Exercise 9c"""
    # Parameters
    n_joints = 10
    parameter_set = [[
        SimulationParameters(
            simulation_duration=10,
            drive=4.5,
            amplitude_gradient = [head, tail],
            phase_lag=2*np.pi,
            turn=0,
            # ...
        )
        for head in np.linspace(0.1,1.0,10)]
        for tail in np.arange(0.1,1.0,10)]
        # for amplitudes in ...
        
    for simulation_i, parameters in enumerate(parameter_set):
        pylog.debug("Param {}".format(simulation_i))
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/simulation9b_{}.npz".format(simulation_i)
        )
       

