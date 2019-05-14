"""Exercise 9b"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import cmc_pylog as pylog
import matplotlib.pyplot as plt


def exercise_9b(world, timestep, reset):
    """Exercise 9b"""
    parameter_set = [[
        SimulationParameters(
            simulation_duration=10,
            drive=4.5,
            amplitude_gradient = [1,1],
            phase_lag=phi,
            amp_factor = amp,
            # ...
        )
        for phi in [ 2*np.pi/(3*10),2*np.pi/10, 3*2*np.pi/10]]
        for amp in np.linspace(0.1,5,25)]
    
    pylog.warning(np.shape(parameter_set))
    phi = [ 2*np.pi/(3*10),2*np.pi/10, 3*2*np.pi/10]
    amp = np.linspace(0.1,5,25)
    energy = np.zeros((len(phi), len(amp)))
    for i in range(0 ,len(phi)):
        for j in range(0,len(amp)):
            reset.reset()
            parameters = parameter_set[i][j]
            run_simulation(
                    world,
                    parameters,
                    timestep,
                    int(1000*parameters.simulation_duration/timestep),
                    logs="./logs/simulation9b_phi{}_amp{}.npz".format(i,amp[j])
                    )

            data = np.load("llogs/simulation9b_phi{}_amp{}.npz".format(i,amp[j]))
            velocity = data["joints"][:,:,1]
            torque = data["joints"][:,:,2]
            energy[i,j] = np.mean(np.trapz(velocity*torque,dx = timestep))
        
    plt.imshow(energy, extent = [2*np.pi/(3*10),6*np.pi/10,0.1,5])
    plt.xlabel('phase lag')
    plt.ylabel('amplitude factor')
    plt.title('Energy estimation')
    plt.colorbar()
    plt.show()

