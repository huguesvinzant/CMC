"""Exercise 9b"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import cmc_pylog as pylog
import plot_results as plot_results

import scipy.integrate as integrate
import scipy.special as special

def exercise_9b(world, timestep, reset):
    """Exercise 9b"""
    """Exercise example"""
    # Parameters
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive=4.5,
            amplitude_gradient = [1,1],
            phase_lag=phi,
            turn=0,
            # ...
        )
        #for phi in [2*np.pi/10, 2*np.pi, np.pi, np.pi/3]
        for phi in [2*np.pi/(3*10), 2*np.pi/10, 3*2*np.pi/10]
        # for amplitudes in ...
        # for ...
    ]
#    parameter_set = [SimulationParameters(simulation_duration = 10,
#                                          drive = 20,
#                                          amplitudes = 10,
#                                          phase_lag=np.zeros(n_joints),
#                                          turn=0)]
#    parameter_set = [
#        SimulationParameters(
#            simulation_duration=10,
#            drive=4.5,
#            amplitude_gradient = grad,
#            phase_lag=2*np.pi/10,
#            turn=0,
#            # ...
#        )
#        #for phi in [2*np.pi/10, 2*np.pi, np.pi, np.pi/3]
#        for grad in [[0.2,1],[0.3,1], [0.5,1], [0.7,1]]
#        # for amplitudes in ...
#        # for ...
#    ]
    # Grid search
    file = open("9b_phaseLag_gridsearch.txt","a+")
    energy = []
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
        data = np.load('logs/simulation9b_{}.npz'.format(simulation_i))
        velocity = np.diff(data["joints"][:,:,0], axis = 0)/timestep
        velocity = np.insert(velocity,0,0,axis = 0)
        torque = data["joints"][:,:,3]
        pylog.info("energyyyyy")
        pylog.info(np.mean(velocity*torque))
        energy.append(np.mean(velocity*torque))
        file.write('\n')
        #file.write(str(np.mean(velocity))+ ' ')
        file.write('trapz mean '+ str(np.mean(np.trapz(velocity*torque,dx = timestep)))+ ' ')
        file.write('trapz std '+ str(np.std(np.trapz(velocity*torque,dx = timestep)))+ ' ')
        file.write('beginning pos ' + str(np.mean(data["links"][1,0,:])))
        file.write('end pos ' + str(np.mean(data["links"][-1,0,:])))
        
        #file.write(str(np.std(velocity*torque))+ '\n')
    file.close()

