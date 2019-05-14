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
        self.amplitude_gradient = parameters.amplitude_gradient
        self.phase_lag = parameters.phase_lag
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

        if parameters.drive_mlr >= d_low and parameters.drive_mlr <= d_high:
            freq_body = 0.2*parameters.drive_mlr + 0.3
        
        d_high = 3.0
        
        if parameters.drive_mlr >= d_low and parameters.drive_mlr <= d_high:
            freq_limb = 0.2*parameters.drive_mlr

        freqs[:20] = freq_body
        freqs[20:] = freq_limb
        
        self.freqs = freqs

        #pylog.warning("Coupling weights must be set")
        pylog.warning(self.freqs)    

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        
        
        matrix = np.zeros([24,24])

        np.fill_diagonal(matrix[1:20], 10)
        np.fill_diagonal(matrix[:,1:20], 10)
        #the oscillators 10 and 11 are not connected in either direction
        matrix[9,10] = 0
        matrix[10,9] = 0
        
        np.fill_diagonal(matrix[10:20], 10)
        np.fill_diagonal(matrix[:,10:20], 10)
        
        
        matrix[20,21:23] = 10
        matrix[21:23,20] = 10
        matrix[23,21:23] = 10
        matrix[21:23,23] = 10
        
        matrix[0:5,20] = 30
        matrix[5:10,21] = 30
        matrix[10:15,22] = 30
        matrix[15:20,23] = 30
        
        self.coupling_weights = matrix
             
        #pylog.warning("Coupling weights must be set")
        
    def set_phase_bias(self, parameters):
        """Set phase bias"""
        
        matrix = np.zeros([24,24])

        upwards_weights = self.phase_lag
        weights = np.pi
        
        np.fill_diagonal(matrix[1:20], upwards_weights)
        np.fill_diagonal(matrix[:,1:20], -upwards_weights)
        #the oscillators 10 and 11 are not connected in either direction
        matrix[9,10] = 0
        matrix[10,9] = 0
        
        np.fill_diagonal(matrix[10:20], weights)
        np.fill_diagonal(matrix[:,10:20], weights)

        
        matrix[20,21:23] = np.pi
        matrix[21:23,20] = np.pi
        matrix[23,21:23] = np.pi
        matrix[21:23,23] = np.pi
        
        matrix[0:5,20] = np.pi
        matrix[5:10,21] = np.pi
        matrix[10:15,22] = np.pi
        matrix[15:20,23] = np.pi
        
        self.phase_bias = matrix
        
        #pylog.warning("Phase bias must be set")

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        
        self.rates = 2*np.ones(self.n_oscillators)
        
        #pylog.warning("Convergence rates must be set")

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        
        nominal_amplitudes = np.zeros(self.n_oscillators)
        
        amp_body = 0
        amp_limb = 0
        
        d_high = 5.0
        d_low = 1.0

        if parameters.drive_mlr >= d_low and parameters.drive_mlr <= d_high:
            amp_body = 0.065*parameters.drive_mlr + 0.196
        
        d_high = 3.0
        
        if parameters.drive_mlr >= d_low and parameters.drive_mlr <= d_high:
            amp_limb = 0.131*parameters.drive_mlr + 0.131
            
        gradient = np.linspace(self.amplitude_gradient[0],self.amplitude_gradient[1], 10)
        pylog.info(np.shape(gradient))
        gradient_ = np.concatenate((gradient, gradient))
        pylog.info(np.shape(gradient_))
        nominal_amplitudes[:20] = amp_body*gradient_
        nominal_amplitudes[20:] = amp_limb
        
        
        self.nominal_amplitudes = nominal_amplitudes
        
        #pylog.warning("Nominal amplitudes must be set")
        

