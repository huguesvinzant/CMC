"""Robot parameters"""

import numpy as np
import cmc_pylog as pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def set_frequencies(self, parameters):
        """Set frequencies"""
        
        freqs = np.ones(self.n_oscillators)
        
        freq_body = 0
        freq_limb = 0
        
        d_high = 5.0
        d_low = 1.0

        if parameters.drive_mlr > d_low and parameters.drive_mlr < d_high:
            freq_body = 0.2*parameters.drive_mlr + 0.3
        
        d_high = 3.0
        
        if parameters.drive_mlr > d_low and parameters.drive_mlr < d_high:
            freq_limb = 0.2*parameters.drive_mlr

        freqs[:20] = freq_body
        freqs[20:] = freq_limb
        
        self.freqs = freqs

        #pylog.warning("Coupling weights must be set")

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        
        matrix = np.zeros([24,24])
        body_weights = 10*np.ones(19)
        np.fill_diagonal(matrix[1:], body_weights)
        np.fill_diagonal(matrix[:,1:], body_weights)
        
        np.fill_diagonal(matrix[10:], body_weights[:10])
        np.fill_diagonal(matrix[:,10:], body_weights[:10])
        
        np.fill_diagonal(matrix[21:], body_weights[:3])
        np.fill_diagonal(matrix[:,21:], body_weights[:3])
        
        matrix[22][20] = 10
        matrix[23][21] = 10
        
        matrix[20][0:5] = 30
        matrix[21][5:10] = 30
        matrix[22][10:15] = 30
        matrix[23][15:20] = 30
        
        self.coupling_weights = matrix
             
        #pylog.warning("Coupling weights must be set")

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        
        matrix = np.zeros([24,24])
        upwards_weights = 2*np.pi/8*np.ones(19)
        weights = np.pi*np.ones(10)
        
        np.fill_diagonal(matrix[1:], -upwards_weights)
        np.fill_diagonal(matrix[:,1:], upwards_weights)
        
        np.fill_diagonal(matrix[10:], weights)
        np.fill_diagonal(matrix[:,10:], weights)
        
        np.fill_diagonal(matrix[21:], weights[:3])
        np.fill_diagonal(matrix[:,21:], weights[:3])
        
        matrix[22][20] = np.pi
        matrix[23][21] = np.pi
        
        matrix[20][0:5] = np.pi
        matrix[21][5:10] = np.pi
        matrix[22][10:15] = np.pi
        matrix[23][15:20] = np.pi
        
        self.phase_bias = matrix
        
        #pylog.warning("Phase bias must be set")

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        
        self.rates = 20*np.ones(self.n_oscillators)
        
        #pylog.warning("Convergence rates must be set")

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        
        nominal_amplitudes = np.zeros(self.n_oscillators)
        
        amp_body = 0
        amp_limb = 0
        
        d_high = 5.0
        d_low = 1.0

        if parameters.drive_mlr > d_low and parameters.drive_mlr < d_high:
            amp_body = 0.065*parameters.drive_mlr + 0.196
        
        d_high = 3.0
        
        if parameters.drive_mlr > d_low and parameters.drive_mlr < d_high:
            amp_limb = 0.131*parameters.drive_mlr + 0.131

        nominal_amplitudes[:20] = amp_body
        nominal_amplitudes[20:] = amp_limb
        
        self.nominal_amplitudes = nominal_amplitudes
        
        #pylog.warning("Nominal amplitudes must be set")

