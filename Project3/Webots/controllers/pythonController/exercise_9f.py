"""Exercise 9f"""

import numpy as np
import math
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters

import matplotlib.pyplot as plt


def exercise_9f(world, timestep, reset):
    """Exercise 9f"""
#    parameters = [SimulationParameters(
#            simulation_duration=20, 
#            drive=1.5, 
#            amplitude_gradient = [0.2,1], 
#            phase_lag=2*np.pi/10, 
#            turn=[1, 'Right'],
#            offset = phi
#            # ...
#        ) for phi in [2*np.pi, np.pi, np.pi/2, np.pi/4, 2*np.pi/10]
#    ]
#    
#    speed = np.zeros(5)
#    
#    for simulation_i, parameters in enumerate(parameters):
#            reset.reset()
#            run_simulation(
#                world,
#                parameters,
#                timestep,
#                int(1000*parameters.simulation_duration/timestep),
#                logs="./logs/simulation9f_{}.npz".format(simulation_i)
#            )
#            data = np.load("logs/simulation9f_{}.npz".format(simulation_i))
#            velocity = np.diff(data["joints"][:,:,0], axis = 0)/timestep
#            velocity = np.insert(velocity,0,0,axis = 0)
#            speed[simulation_i] = np.mean(velocity)
#    
#    plt.plot([2*math.pi, math.pi, math.pi/2, math.pi/4, 2*math.pi/10], speed)
#    plt.xlabel('phase offset')
#    plt.ylabel('mean velocity')
#    plt.title('Velocity as a function of phase offset')
#    plt.show()
    
    
    parameters = [SimulationParameters(
            simulation_duration=20, 
            drive=1.5, 
            amplitude_gradient = [0.2,1], 
            phase_lag=2*np.pi/10, 
            turn=[1, 'Right'],
            offset = np.pi/2, #optimal phase lag
            amp_factor = amp
            # ...
        ) 
        for amp in np.linspace(0,2.0,20)
    ]
    
    speed = np.zeros(20)
    
    for simulation_i, parameters in enumerate(parameters):
            reset.reset()
            run_simulation(
                world,
                parameters,
                timestep,
                int(1000*parameters.simulation_duration/timestep),
                logs="./logs/simulation9f_2{}.npz".format(simulation_i)
            )
            data = np.load("logs/simulation9f_2{}.npz".format(simulation_i))
            velocity = np.diff(data["joints"][:,:,0], axis = 0)/timestep
            velocity = np.insert(velocity,0,0,axis = 0)
            speed[simulation_i] = np.mean(velocity)
    
    plt.plot([2*math.pi, math.pi, math.pi/2, math.pi/4, 2*math.pi/10], speed)
    plt.xlabel('amplitude')
    plt.ylabel('mean velocity')
    plt.title('Velocity as a function of oscillation amplitude')
    plt.show()
