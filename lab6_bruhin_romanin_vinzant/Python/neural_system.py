import cmc_pylog as pylog
import numpy as np

from system_parameters import NetworkParameters


class NeuralSystem(object):
    """Leaky integrator neuron model"""

    def __init__(self, params=NetworkParameters()):
        super(NeuralSystem, self).__init__()
        self.ext_in = np.zeros((4, 1))
        self.params = params
        # Extract parameters
        self.tau, self.D, self.b, self.w, self.exp = (params.tau,
                                                      params.D,
                                                      params.b,
                                                      params.w,
                                                      params.exp)

    def external_inputs(self, ext_in):
        """External inputs to the neurons in the network"""
        self.ext_in = ext_in

    def derivative(self, time, state):
        """ Derivative function of a network of 2 leaky integrator neurons
        y is the vector of membrane potentials (variable m in lecture
        equations)
        yd the derivative of the vector of membrane potentials
        """

        # Update the firing rates:
        x = self.n_act(state)

        # Compute the dentritic sums for all neurons
        dend_sum = np.dot(self.w, x) + self.ext_in

        # Compute the membrane potential derivative:
        yd = (dend_sum - state) / self.tau

        return yd

    def n_act(self, val):
        """Activation function for the neuron model"""
        return 1. / (1 + self.exp(-self.D * (val + self.b)))

