"""Run simulation"""

import cmc_pylog as pylog
import numpy as np
from cmc_robot import SalamanderCMC
import plot_results as plot_results
import matplotlib.pyplot as plt


def run_simulation(world, parameters, timestep, n_iterations, logs):
    """Run simulation"""

    # Set parameters
    pylog.info(
        "Running new simulation:\n  {}".format("\n  ".join([
            "{}: {}".format(key, value)
            for key, value in parameters.items()
        ]))
    )

    # Setup salamander control
    salamander = SalamanderCMC(
        world,
        n_iterations,
        logs=logs,
        parameters=parameters
    )


    # Simulation
#    pos =[]
    iteration = 0
    while world.step(timestep) != -1:
        iteration += 1
        if iteration >= n_iterations:
            break
        salamander.step()
#        pos.append(salamander.position_sensors[1])
    
    
#    plot_results.plot_positions(np.arange(0,n_iterations*timestep, timestep),pos)
    # Log data
    pylog.info("Logging simulation data to {}".format(logs))
    salamander.log.save_data()

