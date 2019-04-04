""" Pendulum """

import numpy as np

import cmc_pylog as pylog
from system_parameters import PendulumParameters


class PendulumSystem(object):
    """Pendulum model main class.
    The Pendulum system class consists of all the methods to setup
    and simulate the pendulum dynamics. You need to implement the
    relevant pendulum equations in the following functions.

    #: To create a pendulum object with default pendulum parameters
    >>> pendulum = PendulumSystem()
    #: To create a pendulum object with pre defined parameters
    >>> from system_parameters import PendulumParameters
    >>> parameters = PendulumParameters()
    >>> parameters.L = 0.3 #: Refer PendulumParameters for more info
    >>> pendulum = PendulumSystem(parameters=parameters)
    #: Method to get the first order derivatives of the pendulum
    >>> pendulum = PendulumSystem()
    >>> theta = 0.0
    >>> dtheta = 0.0
    >>> time
    = 0.0
    >>> derivatives = pendulum.pendulum_system(theta, dtheta, time, torque=0.0)
    """

    def __init__(self, parameters=PendulumParameters()):
        """ Initialization """
        super(PendulumSystem, self).__init__()
        self.origin = np.array([0.0, 0.0])
        self.theta = 0.0
        self.dtheta = 0.0
        self.parameters = parameters

    def pendulum_equation(self, theta, dtheta, torque):
        """ Pendulum equation d2theta = -mgL*sin(theta)/I + torque/I

        with:
            - theta: Angle [rad]
            - dtheta: Angular velocity [rad/s]
            - g: Gravity constant [m/s**2]
            - L: Length [m]
            - mass: Mass [kg]
            - I: Inertia [kg-m**2]
            - sin: np.sin
        """
        # pylint: disable=invalid-name
        g, L, sin, mass, I = (
            self.parameters.g,
            self.parameters.L,
            self.parameters.sin,
            self.parameters.m,
            self.parameters.I
        )

        return (-g * mass * L * sin(theta) + torque) / I

    def pendulum_system(self, time, theta, dtheta, torque):
        """ Pendulum System.
        Accessor method adding pertrubtions."""

        # YOU CAN ADD PERTURBATIONS TO THE PENDULUM MODEL HERE
        if self.parameters.PERTURBATION is True:
            if 1.2 < time < 1.25:
                pylog.warning('Perturbing the pendulum')
                theta = 0.0

        return np.array([
            [dtheta],
            [self.pendulum_equation(theta, dtheta, torque)]  # d2theta
        ])[:, 0]

    def pose(self):
        """Compute the full pose of the pendulum.

        Returns:
        --------
        pose: np.array
            [origin, center-of-mass]"""
        return np.array(
            [self.origin,
             self.origin + self.link_pose()])

    def link_pose(self):
        """ Position of the pendulum center of mass.

        Returns:
        --------
        link_pose: np.array
            Returns the current pose of pendulum COM"""

        return self.parameters.L * np.array([
            np.sin(self.theta),
            -np.cos(self.theta)])

    @property
    def state(self):
        """ Get the pendulum state  """
        return [self.theta, self.dtheta]

    @state.setter
    def state(self, value):
        """"Set the state of the pendulum.

        Parameters:
        -----------
        value: np.array
            Position and Velocity of the pendulum"""

        self.theta = value[0]
        self.dtheta = value[1]

