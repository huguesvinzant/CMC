"""Oscillator network ODE"""

import numpy as np
import cmc_pylog as pylog

from solvers import euler, rk4


def phases_ode(time, phases, freqs, coupling_weights, phases_desired):
    """Network phases ODE"""
    return np.zeros_like(phases)


def amplitudes_ode(time, amplitudes, rate, amplitudes_desired):
    """Network amplitudes ODE"""
    return np.zeros_like(amplitudes)


def motor_output(phases_left, phases_right, amplitudes_left, amplitudes_right):
    """Motor output"""
    return np.zeros_like(amplitudes_left)


class ODESolver(object):
    """ODE solver with step integration"""

    def __init__(self, ode, timestep, solver=rk4):
        super(ODESolver, self).__init__()
        self.ode = ode
        self.solver = solver
        self.timestep = timestep
        self._time = 0

    def integrate(self, state, *parameters):
        """Step"""
        dstate = self.solver(
            self.ode,
            self.timestep,
            self._time,
            state,
            *parameters
        )
        self._time += self.timestep
        return dstate

    def time(self):
        """Time"""
        return self._time


class PhaseEquation(ODESolver):
    """Phase ODE equation"""

    def __init__(self, timestep, freqs, phase_lag):
        super(PhaseEquation, self).__init__(phases_ode, timestep, euler)
        self.n_joints = 10
        self.phases = 1e-4*np.random.ranf(2*self.n_joints)
        self.freqs = np.zeros(2*self.n_joints)
        self.coupling_weights = np.zeros([2*self.n_joints, 2*self.n_joints])
        self.phases_desired = np.zeros([2*self.n_joints, 2*self.n_joints])
        self.set_parameters(freqs, phase_lag)

    def set_parameters(self, freqs, phase_lag):
        """Set parameters of the network"""

        # Set coupling weights
        pylog.warning("Coupling weights must be set")

        # Set desired phases
        pylog.warning("Desired phases must be set")

    def step(self):
        """Step"""
        self.phases += self.integrate(
            self.phases,
            self.freqs,
            self.coupling_weights,
            self.phases_desired
        )


class AmplitudeEquation(ODESolver):
    """Amplitude ODE equation"""

    def __init__(self, timestep, amplitudes, turn):
        super(AmplitudeEquation, self).__init__(
            amplitudes_ode, timestep, euler)
        self.n_joints = 10
        self.amplitudes = np.zeros(2*self.n_joints)
        self.rates = np.zeros(2*self.n_joints)
        self.amplitudes_desired = np.zeros(2*self.n_joints)
        self.set_parameters(amplitudes, turn)

    def set_parameters(self, amplitudes, turn):
        """Set parameters of the network"""

        # Set convergence rates
        pylog.warning("Convergence rates must be set")

        # Set desired amplitudes
        pylog.warning("Desired amplitudes must be set")

    def step(self):
        """Step"""
        self.amplitudes += self.integrate(
            self.amplitudes,
            self.rates,
            self.amplitudes_desired
        )


class SalamanderNetwork(object):
    """Salamander oscillator network"""

    def __init__(self, timestep, freqs, amplitudes, phase_lag, turn):
        super(SalamanderNetwork, self).__init__()
        # Phases
        self.phase_equation = PhaseEquation(
            timestep,
            freqs,
            phase_lag
        )
        # Amplitude
        self.amplitude_equation = AmplitudeEquation(
            timestep,
            amplitudes,
            turn
        )

    def step(self):
        """Step"""
        self.phase_equation.step()
        self.amplitude_equation.step()

    def get_motor_position_output(self):
        """Get motor position"""
        return motor_output(
            self.phase_equation.phases[:10],
            self.phase_equation.phases[10:],
            self.amplitude_equation.amplitudes[:10],
            self.amplitude_equation.amplitudes[10:]
        )

