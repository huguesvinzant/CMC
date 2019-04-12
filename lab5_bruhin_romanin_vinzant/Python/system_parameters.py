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

        Spring Parameters:
            - k1 : Spring constant of spring 1 [N/rad]
            - k2 : Spring constant of spring 2 [N/rad]

            - s_theta_ref1 : Spring 1 reference angle [rad]
            - s_theta_ref2 : Spring 2 reference angle [rad]

         Damper Parameters:
            - b1 : Damping constant damper 1 [N-s/rad]
            - b2 : Damping constant damper 2 [N-s/rad]

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

        # Spring Parameters
        - k1 = 10.
        - k2 = 10.
        - s_theta_ref1 =  0.0
        - s_theta_ref2 =  0.0

        # Damping Parameters
        - b1 = 0.5
        - b2 = 0.5


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

    Spring parameters:
        k1: 10.0 [N/rad]
        k2: 10.0 [N/rad]
        s_theta_ref1: 0.0 [rad]
        s_theta_ref2: 0.0 [rad]

    Damping parameters:
        b1: 0.5 [N-s/rad]
        b2: 0.5 [N-s/rad]

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
        self.units['k1'] = 'N/rad'
        self.units['k2'] = 'N/rad'
        self.units['s_theta_ref1'] = 'rad'
        self.units['s_theta_ref2'] = 'rad'
        self.units['b1'] = 'N-s/rad'
        self.units['b2'] = 'N-s/rad'

        # Initialize parameters
        self.parameters = {
            'g': 9.81, 'm': 1., 'L': 1., 'I': 0.0, 'sin': np.sin, 'k1': 0.,
            'k2': 0., 's_theta_ref1': 0., 's_theta_ref2': 0.,
            'b1': 0., 'b2': 0.}

        # Pendulum parameters
        self.g = kwargs.pop("g", 9.81)  # Gravity constant
        self.m = kwargs.pop("m", 1.)  # Mass
        self.L = kwargs.pop("L", 1.)  # Length
        self.sin = kwargs.pop("sin", np.sin)  # Sine function
        # Spring parameters
        self.k1 = kwargs.pop(
            "k1", 10.)  # Spring constant of Spring 1
        self.k2 = kwargs.pop(
            "k2", 10.)  # Spring constant of Spring 2
        self.s_theta_ref1 = kwargs.pop(
            "s_theta_ref1", 0.0)  # Spring 1 reference angle
        self.s_theta_ref2 = kwargs.pop(
            "s_theta_ref2", 0.0)  # Spring 2 reference angle
        # Damping parameters
        self.b1 = kwargs.pop(
            "b1", 0.5)  # Damping constant of Damper 1
        self.b2 = kwargs.pop(
            "b2", 0.5)  # Damping constant of Damper 2

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
    def k1(self):
        """ Get the value of spring constant for spring 1. [N/rad]
        Default is -10.0"""
        return self.parameters['k1']

    @k1.setter
    def k1(self, value):
        """ Keyword Arguments:
        value -- Set the value of spring constant for spring 1 [N/rad] """
        if (value < 0.0):
            pylog.warning('Setting bad spring constant. Should be positive!')
        else:
            self.parameters['k1'] = value
        pylog.info(
            'Changed spring constant of spring 1 to {} [N/rad]'.format(self.parameters['k1']))

    @property
    def k2(self):
        """ Get the value of spring constant for spring 2. [N/rad]
        Default is -10.0"""
        return self.parameters['k2']

    @k2.setter
    def k2(self, value):
        """ Keyword Arguments:
        value -- Set the value of spring constant for spring 2[N/rad] """
        if (value < 0.0):
            pylog.warning('Setting bad spring constant. Should be positive!')
        else:
            self.parameters['k2'] = value
        pylog.info(
            'Changed spring constant of spring 1 to {} [N/rad]'.format(self.parameters['k2']))

    @property
    def s_theta_ref1(self):
        """ Get the value of spring 1 reference angle. [rad]
        Default is 0.0"""
        return self.parameters['s_theta_ref1']

    @s_theta_ref1.setter
    def s_theta_ref1(self, value):
        """ Keyword Arguments:
        value -- Set the value of spring 1 reference angle [rad] """
        self.parameters['s_theta_ref1'] = value
        pylog.info(
            'Changed spring 1 reference angle to {} [rad]'.format(
                self.parameters['s_theta_ref1']))

    @property
    def s_theta_ref2(self):
        """ Get the value of spring 2 reference angle. [rad]
        Default is 0.0"""
        return self.parameters['s_theta_ref2']

    @s_theta_ref2.setter
    def s_theta_ref2(self, value):
        """ Keyword Arguments:
        value -- Set the value of spring 2 reference angle [rad] """
        self.parameters['s_theta_ref2'] = value
        pylog.info(
            'Changed spring 2 reference angle to {} [rad]'.format(
                self.parameters['s_theta_ref2']))

    @property
    def b1(self):
        """ Get the value of damping constant for damper 1. [N-s/rad]
        Default is 0.5"""
        return self.parameters['b1']

    @b1.setter
    def b1(self, value):
        """ Keyword Arguments:
        value -- Set the value of damping constant for damper 1. [N-s/rad] """
        if (value < 0.0):
            pylog.warning('Setting bad damping values. Should be positive!')
        else:
            self.parameters['b1'] = value
        pylog.info(
            'Changed damping constant for damper 1 to {} [N-s/rad]'.format(self.parameters['b1']))

    @property
    def b2(self):
        """ Get the value of damping constant for damper 2. [N-s/rad]
        Default is 0.5"""
        return self.parameters['b2']

    @b2.setter
    def b2(self, value):
        """ Keyword Arguments:
        value -- Set the value of damping constant for damper 2. [N-s/rad] """
        if (value < 0.0):
            pylog.warning('Setting bad damping values. Should be positive!')
        else:
            self.parameters['b2'] = value
        pylog.info(
            'Changed damping constant for damper 2 to {} [N-s/rad]'.format(self.parameters['b2']))

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
        self.parameters['l_opt'] = kwargs.pop('l_opt', 0.11)
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
        """ Muscle fiber pennation angle  """
        return self.parameters['pennation']

    @pennation.setter
    def pennation(self, value):
        """ Keyword Arguments:
            value -- Muscle fiber pennation angle """
        self.parameters['pennation'] = value

    def showParameters(self):
        return self.msg(self.parameters, self.units)


class MassParameters(SystemParameters):
    """ Mass parameters

    with:
        Mass Parameters:
            - g : Mass system gravity [m/s**2]
            - mass : Mass of the object [kg]

    Examples:

        >>> mass_parameters = MassParameters(g = 9.81, mass = 9.81)

    Note that not giving arguments to instanciate the object will result in the
    following default values:
        # Mass Parameters
        - g = 9.81
        - mass = 10.

    These parameter variables can then be called from within the class using
    for example:

        To assign a new value to the object variable from within the class:

        >>> self.g = 10.0 # Reassign gravity constant

        To assign to another variable from within the class:

        >>> example_g = self.g

    You can display the parameters using:

    >>> mass_parameters = MassParameters()
    >>> print(mass_parameters,showParameters())
    Mass parameters :
            g : 9.81 [m/s**2]
            mass : 10. [kg]

    Or using biolog:

    >>> mass_parameters = MassParameters()
    >>> biolog.info(mass_parameters.showParameters())
    """

    def __init__(self, **kwargs):
        super(MassParameters, self).__init__('Mass')
        self.parameters = {}
        self.units = {}

        self.units['g'] = 'm/s**2'
        self.units['mass'] = 'kg'

        self.parameters['g'] = kwargs.pop('g', 9.81)
        self.parameters['mass'] = kwargs.pop('mass', 10.)

    @property
    def g(self):
        """ Get the value of gravity in mass   """
        return self.parameters['g']

    @g.setter
    def g(self, value):
        """ Keyword Arguments:
            value --  Set the value of gravity"""
        self.parameters["g"] = value

    @property
    def mass(self):
        """Get the value of mass in the mass system  """
        return self.parameters["mass"]

    @mass.setter
    def mass(self, value):
        """ Keyword Arguments:
            value --  Set the value of mass"""
        if value <= 0.00001:
            biolog.error(
                "Mass you are trying to set is too low!. Setting to 1.")
            value = 1.0
        self.parameters["mass"] = value

    def showParameters(self):
        return self.msg(self.parameters, self.units)


if __name__ == '__main__':
    P = PendulumParameters(g=9.81, L=1.)
    pylog.debug(P.showParameters())

    M = MuscleParameters()
    pylog.debug(M.showParameters())

    Mass = MassParameters()
    pylog.debug(Mass.showParameters())

