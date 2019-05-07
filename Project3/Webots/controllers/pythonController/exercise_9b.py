"""Exercise 9b"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import cmc_pylog as pylog

def exercise_9b(world, timestep, reset):
    """Exercise 9b"""
    """Exercise example"""
    # Parameters
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive=drive,
            amplitudes=[1, 2, 3],
            phase_lag=np.zeros(n_joints),
            turn=0,
            # ...
        )
        for drive in np.linspace(1, 10, 100)
        # for amplitudes in ...
        # for ...
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        pylog.debug("Param {}".format(simulation_i))
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/simulation_{}.npz".format(simulation_i)
        )



