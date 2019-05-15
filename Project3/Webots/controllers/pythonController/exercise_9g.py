"""Exercise 9g"""

# from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9g(world, timestep, reset):
    """Exercise 9g"""
    
    parameters = SimulationParameters(
            simulation_duration=20, 
            amplitude_gradient = [0.2,1], 
            phase_lag=2*np.pi/10, 
            turn=[1, 'Right'],
            offset = np.pi, #optimal phase lag
            amp_factor = amp
            ) 

