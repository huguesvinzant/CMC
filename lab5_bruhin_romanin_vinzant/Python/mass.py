""" Simple Mass system. """
import numpy as np
import cmc_pylog as pylog
from system_parameters import MassParameters


class Mass(object):
    """Simple mass system.
    """

    def __init__(self, parameters=None):
        super(Mass, self).__init__()
        if parameters is None:
            pylog.warning('Setting default parameters to mass model.')
            self.parameters = MassParameters()
        else:
            self.parameters = parameters

    def mass_equation(self, state, time, *args):
        """ Mass equation. xdd = g - F/m"""
        #: Unwrap the parameters
        mass, gravity = (self.parameters.mass, self.parameters.g)
        force = args[0]
        return gravity - force / mass

    def dxdt(self, state, time, *args):
        """ Muscle-Mass System"""
        velocity = state[1]
        return np.array(
            [velocity,
             self.mass_equation(state, time, *args)])  # xdd)

