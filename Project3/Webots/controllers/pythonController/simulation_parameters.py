"""Simulation parameters"""
import numpy as np

class SimulationParameters(dict):
    """Simulation parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 10
        self.n_legs_joints = 4
        self.simulation_duration = 30
        self.phase_lag = (2*np.pi/10)
        self.amplitude_gradient = [0.2,1]
        self.turn = [1, 'Right']
        # Feel free to add more parameters (ex: MLR drive)
        self.drive_mlr = 4.5
        self.offset = np.pi
    
        # ...
        # Update object with provided keyword arguments
        self.update(kwargs)  # NOTE: This overrides the previous declarations

