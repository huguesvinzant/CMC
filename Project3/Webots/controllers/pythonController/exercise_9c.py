"""Exercise 9c"""

import numpy as np
import matplotlib.pyplot as plt
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import cmc_pylog as pylog
from tempfile import TemporaryFile



def exercise_9c(world, timestep, reset):
    """Exercise 9c"""
    # Parameters
    outfile = TemporaryFile()

    parameter_set = [[
        SimulationParameters(
            simulation_duration=20,
            drive=4.5,
            amplitude_gradient = [head, tail],
            phase_lag=2*np.pi/10,
            # ...
        )
        for head in np.linspace(0.1,1.5,15)]
        for tail in np.linspace(0.1,1.5,15)]
        # for amplitudes in ...
    
    pylog.warning(np.shape(parameter_set))
    head = np.linspace(0.1,1.5,15)
    tail = np.linspace(0.1,1.5,15)
    pylog.warning(head)
    energy = np.zeros((len(head), len(tail)))
    for i in range(0 ,len(head)):
        for j in range(0,len(tail)):
            reset.reset()
            parameters = parameter_set[i][j]
            run_simulation(
                    world,
                    parameters,
                    timestep,
                    int(1000*parameters.simulation_duration/timestep),
                    logs="./logs/simulation9c_more_head{}_tail{}.npz".format(head[i],tail[j])
                    )

            data = np.load("logs/simulation9c_more_head{}_tail{}.npz".format(head[i],tail[j]))
            velocity = np.diff(data["joints"][:,:,0], axis = 0)/timestep
            velocity = np.insert(velocity,0,0,axis = 0)
            torque = data["joints"][:,:,2]
            energy[i,j] = np.mean(np.trapz(velocity*torque,dx = timestep))
      
    np.save(outfile, energy)
    outfile.seek(0) #for when you want to open it again

    plt.imshow(energy, extent = [0.1,1,0.1,1])
    plt.xlabel('head factor')
    plt.ylabel('tail factor')
    plt.title('Energy estimation')
    plt.colorbar()
    plt.show()