"""Exercise 9b"""

import numpy as np
import math
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import cmc_pylog as pylog
import matplotlib.pyplot as plt
from tempfile import TemporaryFile

def exercise_9b(world, timestep, reset):
    """Exercise 9b"""
    
    outfile = TemporaryFile()
    
    parameter_set = [[
        SimulationParameters(
            simulation_duration=10,
            drive=4.5,
            amplitude_gradient = [1,1],
            phase_lag=phi,
            amp_factor = amp,
            # ...
        )
        for phi in np.array([ 2*np.pi/(3*10), 2*np.pi/(2*10),2*np.pi/10,2*2*np.pi/10, 3*2*np.pi/10])]
        for amp in np.linspace(0,1,20)]
    
    pylog.warning(np.shape(parameter_set))
    phi = np.array([ 2*np.pi/(3*10), 2*np.pi/(2*10),2*np.pi/10,2*2*np.pi/10, 3*2*np.pi/10])
    amp = np.linspace(0,1,20)
    energy = np.zeros_like(parameter_set)
    for ind, angle in enumerate(phi):
        for j in range(0, len(amp)):
            pylog.warning('angle index '+str(ind))
            pylog.warning('amp index '+str(j) + ' ampfactor used: ' + str(amp[j]))
            reset.reset()
            parameters = parameter_set[j][ind]
            run_simulation(
                    world,
                    parameters,
                    timestep,
                    int(1000*parameters.simulation_duration/timestep),
                    logs="./logs/simulation9b_phi{}_amp{}.npz".format(ind,j)
                    )

            data = np.load("logs/simulation9b_phi{}_amp{}.npz".format(ind,j))
            velocity = np.diff(data["joints"][:,:,0], axis = 0)/timestep
            velocity = np.insert(velocity,0,0,axis = 0)
            torque = data["joints"][:,:,2]
            energy[j,ind] = np.mean(np.trapz(velocity*torque,dx = timestep))
            pylog.info('Energy: '+ str(energy[j,ind]))
        
    np.save(outfile, energy)

    #outfile.seek(0) #for when you want to open it again
    plt.imshow(energy.T, extent = [2*math.pi/(3*10),6*math.pi/10,0,5])
    plt.xlabel('phase lag')
    plt.ylabel('amplitude factor')
    plt.title('Energy estimation')
    plt.colorbar()
    plt.show()
    with open('energy9b.txt','wb') as f:
        for line in energy:
            np.savetxt(f, line, fmt='%.2f')

