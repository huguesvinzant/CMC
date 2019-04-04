""" Lab 4 : System Parameters """

import numpy as np
import cmc_pylog as pylog


class SystemParameters(object):
    """Parent class providing main attributes for other sub system parameters.
    """

    def __init__(self, name='System'):
        super(SystemParameters, self).__init__()
        self. name = name

    def showParameters(self):
        raise NotImplementedError()

    def msg(self, parameters, units, endl="\n" + 4 * " "):
        """ Message """
        to_print = ("{} parameters : ".format(self.name)) + endl
        for param in parameters:
            to_print += ("{} : {} [{}]".format(param,
                                               parameters[param], units[param])) + endl
        return to_print


class PendulumParameters(SystemParameters):
    """ Pendulum parameters

    with:
        Pendulum Parameters:
            - g: Gravity constant [m/s**2]
            - m: Mass [kg]
            - L: Length [m]
            - I: Inerita [kg-m**2]
            - sin: Sine function

    Examples:

        >>> pendulum_parameters = PendulumParameters(g=9.81, L=0.1)

    Note that not giving arguments to instanciate the object will result in the
    following default values:
        # Pendulum Parameters
        - g = 9.81
        - m = 1.
        - L = 1.
        - I = 1.
        - sin = np.sin


    These parameter variables can then be called from within the class using
    for example:

        To assign a new value to the object variable from within the class:

        >>> self.g = 9.81 # Reassign gravity constant

        To assign to another variable from within the class:

        >>> example_g = self.g

    To call the parameters from outside the class, such as after instatiation
    similarly to the example above:

        To assign a new value to the object variable from outside the class:

        >>> pendulum_parameters = SystemParameters(L=1.0)
        >>> pendulum_parameters.L = 0.3 # Reassign length

        To assign to another variable from outside the class:

        >>> pendulum_parameters = SystemParameters()
        >>> example_g = pendulum_parameters.g # example_g = 9.81

    You can display the parameters using:

    >>> pendulum_parameters = SystemParameters()
    >>> print(pendulum_parameters.showParameters())
    Pendulum parameters:
        g: 9.81 [m/s**2]
        m: 1. [kg]
        L: 1.0 [m]
        I: 1.0 [kg-m**2]
        sin: <ufunc 'sin'>

    Or using pylog:

    >>> pendulum_parameters = SystemParameters()
    >>> pylog.info(system_parameters.showParameters())
    """

    def __init__(self, **kwargs):
        super(PendulumParameters, self).__init__('Pendulum')

        self.parameters = {}
        self.units = {}

        self.units['g'] = 'N-m/s2'
        self.units['m'] = 'kg'
        self.units['L'] = 'm'
        self.units['I'] = 'kg-m**2'
        self.units['sin'] = ''
        self.units['PERTURBATION'] = 'bool'

        # Initialize parameters
        self.parameters = {
            'g': 9.81, 'm': 1., 'L': 1., 'I': 0.0, 'sin': np.sin,
            'PERTURBATION': False}

        # Pendulum parameters
        self.g = kwargs.pop("g", 9.81)  # Gravity constant
        self.m = kwargs.pop("m", 1.)  # Mass
        self.L = kwargs.pop("L", 1.)  # Length
        self.sin = kwargs.pop("sin", np.sin)  # Sine function
        self.PERTURBATION = kwargs.pop("PERTURBATION", False)  # Perturbation

        pylog.info(self)
        return

    @property
    def g(self):
        """ Get the value of gravity in the system. [N-m/s2]
        Default is 9.81 """
        return self.parameters['g']

    @g.setter
    def g(self, value):
        """ Keyword Arguments:
        value -- Set the value of gravity [N-m/s2] """
        self.parameters['g'] = value
        pylog.info(
            'Changed gravity to {} [N-m/s2]'.format(self.parameters['g']))

    @property
    def m(self):
        """ Get the mass of the pendulum."""
        return self.parameters['m']

    @m.setter
    def m(self, value):
        """
        Set the mass of the pendulum.
        Setting/Changing mass will automatically recompute the inertia.
        """
        self.parameters['m'] = value
        # ReCompute inertia
        # Inertia = m*l**2
        self.I = self.parameters['m']*self.L**2
        pylog.debug(
            'Changed pendulum mass to {} [kg]'.format(self.m))

    @property
    def I(self):
        """ Get the inertia of the pendulum [kg-m**2]  """
        return self.parameters['I']

    @I.setter
    def I(self, value):
        """ Set the value of the pendulum inertia """
        self.parameters['I'] = value

    @property
    def L(self):
        """ Get the value of pendulum length. [m]
        Default is 1.0"""
        return self.parameters['L']

    @L.setter
    def L(self, value):
        """ Keyword Arguments:
        value -- Set the value of pendulum's length [m] """
        self.parameters['L'] = value
        # ReCompute inertia
        # Inertia = m*l**2
        self.I = self.m*self.parameters['L']**2
        pylog.debug(
            'Changed pendulum length to {} [m]'.format(self.L))

    @property
    def sin(self):
        """ Get the sine function."""
        return self.parameters['sin']

    @sin.setter
    def sin(self, value):
        """ Set the sine function to be used. """
        self.parameters['sin'] = value

    @property
    def PERTURBATION(self):
        """ Enable/Disable pendulum perturbation.  """
        return self.parameters['PERTURBATION']

    @PERTURBATION.setter
    def PERTURBATION(self, value):
        """Keyword Arguments:
           value -- Enable/Disable pendulum perturbation"""
        self.parameters['PERTURBATION'] = value

    def showParameters(self):
        return self.msg(self.parameters, self.units)


class MuscleParameters(SystemParameters):
    """ Muscle parameters

    with:
        Muscle Parameters:
            - l_slack : Tendon slack length [m]
            - l_opt : Contracticle element optimal fiber length [m]
            - f_max : Maximum force produced by the muscle [N]
            - v_max : Maximum velocity of the contracticle element [m/s]
            - pennation : Fiber pennation angle

    Examples:

        >>> muscle_parameters = MuscleParameters(l_slack=0.2, l_opt=0.1)

    Note that not giving arguments to instanciate the object will result in the
    following default values:
        # Muscle Parameters
        - l_slack = 0.13
        - l_opt = 0.11
        - f_max = 1500
        - v_max = 1.2
        - pennation = 1.

    These parameter variables can then be called from within the class using
    for example:

        To assign a new value to the object variable from within the class:

        >>> self.l_slack = 0.2 # Reassign tendon slack constant

        To assign to another variable from within the class:

        >>> example_l_slack = self.l_slack

    You can display the parameters using:

    >>> muscle_parameters = MuscleParameters()
    >>> print(muscle_parameters,showParameters())
    Muscle parameters :
            f_max : 1500 [N]
            v_max : 1.2 [m/s]
            pennation : 1 []
            l_slack : 0.13 [m]
            l_opt : 0.11 [m]

    Or using biolog:

    >>> muscle_parameters = MuscleParameters()
    >>> biolog.info(muscle_parameters.showParameters())
    """

    def __init__(self, **kwargs):
        super(MuscleParameters, self).__init__('Muscle')
        self.parameters = {}
        self.units = {}

        self.units['l_slack'] = 'm'
        self.units['l_opt'] = 'm'
        self.units['f_max'] = 'N'
        self.units['v_max'] = 'm/s'
        self.units['pennation'] = ''

        self.parameters['l_slack'] = kwargs.pop('l_slack', 0.13)
        self.parameters['l_opt'] = kwargs.pop('l_opt', 0.1)
        self.parameters['f_max'] = kwargs.pop('f_max', 1500)
        self.parameters['v_max'] = kwargs.pop('v_max', -12)
        self.parameters['pennation'] = kwargs.pop('pennation', 1)

    @property
    def l_slack(self):
        """ Muscle Tendon Slack length [m]  """
        return self.parameters['l_slack']

    @l_slack.setter
    def l_slack(self, value):
        """ Keyword Arguments:
            value -- Muscle Tendon Slack Length [m]"""
        self.parameters['l_slack'] = value

    @property
    def l_opt(self):
        """ Muscle Optimal Fiber Length [m]  """
        return self.parameters['l_opt']

    @l_opt.setter
    def l_opt(self, value):
        """ Keyword Arguments:
        value -- Muscle Optimal Fiber Length [m]"""
        self.parameters['l_opt'] = value

    @property
    def f_max(self):
        """ Maximum tendon force produced by the muscle [N]  """
        return self.parameters['f_max']

    @f_max.setter
    def f_max(self, value):
        """ Keyword Arguments:
        value -- Maximum tendon force produced by the muscle [N]"""
        self.parameters['f_max'] = value

    @property
    def v_max(self):
        """ Maximum velocity of the contractile element [m/s]  """
        return self.parameters['v_max']

    @v_max.setter
    def v_max(self, value):
        """ Keyword Arguments:
        value -- Maximum velocity of the contractile element [m/s] """
        self.parameters['v_max'] = value

    @property
    def pennation(self):
        """ Get the value of pennation Muscle fiber pennation angle  """
        return self.parameters['pennation']

    @pennation.setter
    def pennation(self, value):
        """ Keyword Arguments:
            value -- Muscle fiber pennation angle """
        self.parameters['pennation'] = value

    def showParameters(self):
        return self.msg(self.parameters, self.units)


class NetworkParameters(SystemParameters):
    """ Network parameters

    with:
        Network Parameters:
            - tau : Array of time constants for each neuron [s]
            - D : Sigmoid constants for each neuron
            - b : Array of bias for each neuron
            - w : Weight matrix for network connections
            - exp : Exponential function <exp>

    Examples:

        >>> network_parameters = NetworkParameters(tau=[0.02, 0.02, 0.1, 0.1], D=1.)

    Note that not giving arguments to instanciate the object will result in the
    following default values:
        # Neuron Parameters
        - tau = [0.02, 0.02, 0.1, 0.1]
        - D = 1
        - b = [3.0, 3.0, -3.0, -3.0]
        - w = [[0., 1., 1., 1.],
               [1., 0., 1., 1.],
               [1., 1., 0., 1.],
               [1., 1., 1., 0.]]
        - exp = np.exp

    These parameter variables can then be called from within the class using
    for example:

        To assign a new value to the object variable from within the class:

        >>> self.tau[0] = 0.01  # Reassign tendon slack constant

        To assign to another variable from within the class:

        >>> example_tau = self.tau

    You can display the parameters using:

    >>> network_parameters = NetworkParameters()
    >>> print(network_parameters,showParameters())
    Network parameters :
        tau = [0.02, 0.02, 0.1, 0.1]
        D = 1
        b = [3.0, 3.0, -3.0, -3.0]
        w = [[0., 1., 1., 1.],
             [1., 0., 1., 1.],
             [1., 1., 0., 1.],
             [1., 1., 1., 0.]]
        exp = np.exp

    Or using biolog:

    >>> network_parameters = NetworkParameters()
    >>> biolog.info(network_parameters.showParameters())
    """

    def __init__(self, **kwargs):
        super(NetworkParameters, self).__init__('network')
        self.parameters = {}
        self.units = {}

        self.units['tau'] = 's'
        self.units['D'] = '-'
        self.units['b'] = '-'
        self.units['w'] = '-'
        self.units['exp'] = '<exp>'

        # Initialize parameters
        weight_ = np.ones((4, 4))
        np.fill_diagonal(weight_, 0.)
        self.parameters = {
            'tau': np.array([0.02, 0.02, 0.1, 0.1]),
            'D': 1.,
            'b': np.array([3., 3., -3., -3.]),
            'w': weight_,
            'exp': np.exp}

        self.parameters['tau'] = kwargs.pop(
            'tau', np.array([0.02, 0.02, 0.1, 0.1]))
        self.parameters['D'] = kwargs.pop('D', 1.)
        self.parameters['b'] = kwargs.pop('b', np.array([3., 3., -3., -3.]))

        self.parameters['w'] = kwargs.pop(
            'w', weight_)
        self.parameters['exp'] = kwargs.pop('exp', np.exp)

    @property
    def tau(self):
        """ Time constants for neurons in the network  """
        return self.parameters['tau']

    @tau.setter
    def tau(self, value):
        """ Keyword Arguments:
            value -- Time constants for neurons in the network"""
        self.parameters['tau'] = value

    @property
    def D(self):
        """Sigmoid constant  """
        return self.parameters['D']

    @D.setter
    def D(self, value):
        """Keyword Arguments:
           value --  Sigmoid constant """
        self.parameters['D'] = value

    @property
    def b(self):
        """Bias for neurons in the network  """
        return self.parameters['b']

    @b.setter
    def b(self, value):
        """Keyword Arguments:
           value --  Bias for neurons in the network """
        self.parameters['b'] = value

    @property
    def w(self):
        """ weight matrix for the network  """
        return self.parameters['w']

    @w.setter
    def w(self, value):
        """Keyword Arguments:
           value --  weight matrix for the network """
        self.parameters['w'] = value

    def showParameters(self):
        return self.msg(self.parameters, self.units)

    @property
    def exp(self):
        """Exponential"""
        return self.parameters['exp']


if __name__ == '__main__':
    P = PendulumParameters(g=9.81, L=1.)
    pylog.debug(P.showParameters())

    M = MuscleParameters()
    pylog.debug(M.showParameters())

