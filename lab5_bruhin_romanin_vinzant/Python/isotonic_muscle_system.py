""" This contains the methods to simulate isotonic muscle contraction. """

import numpy as np

import cmc_pylog as pylog
from cmcpack import integrate
from mass import Mass
from muscle import Muscle
from system_parameters import MassParameters, MuscleParameters


class IsotonicMuscleSystem(object):
    """System to simulate isotoni muscle system.
    """

    def __init__(self):
        """ Initialization.

        Parameters
        ----------
        None

        Example:
        -------
        >>> isotonic_system = IsotonicMuscleSystem()
        """

        super(IsotonicMuscleSystem, self).__init__()
        self.muscle = None
        self.mass = None

    def add_muscle(self, muscle):
        """Add the muscle to the system.

        Parameters
        ----------
        muscle: <Muscle>
            Instance of muscle model

        Example:
        --------
        >>> from muscle import Muscle
        >>> from system_parameters import MuscleParameters
        >>> muscle = Muscle(MuscleParameters()) #: Default muscle
        >>> isotonic_system = IsotonicMuscleSystem()
        >>> isotonic_system.add_muscle(muscle)
        """
        if self.muscle is not None:
            pylog.warning(
                'You have already added the muscle model to the system.')
            return
        else:
            if muscle.__class__ is not Muscle:
                pylog.error(
                    'Trying to set of type {} to muscle'.format(
                        muscle.__class__))
                raise TypeError()
            else:
                pylog.info('Added new muscle model to the system')
                self.muscle = muscle

    def add_mass(self, mass):
        """Add the mass to the system.

        Parameters
        ----------
        mass: <Mass>
            Instance of mass model

        Example:
        --------
        >>> from mass import Mass
        >>> from system_parameters import MassParameters
        >>> mass = Muscle(MassParameters()) #: Default mass
        >>> isotonic_system = IsotonicMuscleSystem()
        >>> isotonic_system.add_muscle(muscle)
        """
        if self.mass is not None:
            pylog.warning(
                'You have already added the mass model to the system.')
            return
        else:
            if mass.__class__ is not Mass:
                pylog.error(
                    'Trying to set of type {} to mass'.format(mass.__class__))
                raise TypeError()
            else:
                pylog.info('Added new mass model to the system')
                self.mass = mass

    def integrate(self, x0, time, time_step=None, time_stabilize=0.1,
                  stimulation=1.0,
                  load=1.):
        """ Method to integrate the muscle model.

        Parameters:
        ----------
            x0 : <array>
                Initial state of the mass and muscle
                    x0[0] --> activation
                    x0[1] --> contractile length (l_ce)
                    x0[2] --> position of the mass/load
                    x0[3] --> velocity of the mass/load
            time : <array>
                Time vector
            time_step : <float>
                Time step to integrate (Good value is 0.001)
            time_stabilize :<float>
                Time allowed for muscle to settle before quick release
            stimulation : <float>
                Muscle stimulation
            load : <float>
                External load applied to the muscle [kg]


        Returns:
        --------
            result : <Result>
            result.time :
                Time vector
            result.activation :
                Muscle activation state
            result.l_ce :
                Length of contractile element
            result.v_ce :
                Velocity of contractile element
            result.l_mtc :
                Total muscle tendon length
            result.active_force :
                 Muscle active force
            result.passive_force :
                Muscle passive force
            result.tendon_force :
                Muscle tendon force

        Example:
        --------
            >>> import nump as np
            >>> from muscle import Muscle
            >>> from mass import Mass
            >>> from system_parameters import MuscleParameters, MassParameters
            >>> muscle = Muscle(MuscleParameters()) #: Default muscle
            >>> mass = Mass(MassParameters()) #: Default mass
            >>> isotonic_system = IsotonicMuscleSystem()
            >>> isotonic_system.add_muscle(muscle)
            >>> # Initial state
            >>> x0 = [0, isotonic_system.muscle.L_OPT,
                isotonic_system.muscle.L_OPT+isotonic_system.muscle.L_SLACK, 0.0]
            >>> time_step = 0.001
            >>> t_start = 0.0
            >>> t_stop = 0.3
            >>> #: Time
            >>> time = np.arange(t_start, t_stop, time_step)
            >>> time_stabilize = 0.2
            >>> # Args take stimulation and muscle_length as input
            >>> # Set the load to which you want to evaluate
            >>> load = 100 # [kg]
            >>> # Set the muscle stimulation to which you want to evaluate
            >>> muscle_stimulation = 0.5
            >>> args = (muscle_stimulation, load)
            >>> result = isotonic_system.integrate(x0, time, time_step,
                time_stabilize, args)
            >>> # results contain the states and the internal muscle
            >>> # attributes neccessary to complete the exercises

        The above example shows how to run the isotonic condition once.
        In the exercise1.py file you have to use this setup to loop
        over multiple muscle loads/muscle stimulation values to answer
        the questions.
        """

        if time[-1] < .1:
            pylog.error("To short a time to integrate the model "
                        "for quick release experiment!!!")
            raise ValueError()

        if time_step is None:
            time_step = time[1] - time[0]

        if time_stabilize is None:
            pylog.warning("Muscle stabilization time not specified")
            time_stabilize = 0.2

        #: Set the mass attached to the muscle
        self.mass.parameters.mass = load

        #: Integration
        pylog.info(
            "Begin isometric test with load {} and "
            "muscle activation {}".format(load, stimulation))
        #: Instatiate the muscle results container
        self.muscle.instantiate_result_from_state(time)

        #: Integrate the model until stabilization
        for idx, _time in enumerate(time):
            if _time < time_stabilize:
                #: Fixed muscle - Stabilization
                muscle_length = self.muscle.L_OPT + self.muscle.L_SLACK
                res = self.step(x0, [_time, _time+time_step], stimulation,
                                muscle_length)
                x0[:2] = res.state[-1][:2]
            else:
                #: Quick release
                res = self.step(x0, [_time, _time+time_step], stimulation)
                x0 = res.state[-1][:]
                #: Results
                muscle_length = res.state[-1][2]
                self.muscle.generate_result_from_state(idx, _time,
                                                       muscle_length,
                                                       res.state[-1][:])
        # print(_time, self.muscle.Result.tendon_force[-1])
        return self.muscle.Result

    def muscle_mass_system(self, state, time, *args):
        """ Equations for muscle and mass system together. """

        #: Unwrap the args
        stimulation = args[0]
        muscle_length = args[1]

        if muscle_length is None:
            muscle_length = state[2]

        #: Get the muscle force
        muscle_contractile_length = state[1]
        muscle_tendon_length = muscle_length - muscle_contractile_length
        muscle_force = self.muscle.compute_tendon_force(
            muscle_tendon_length)

        muscle_state = self.muscle.dxdt(
            state[:2], time, stimulation, muscle_length)
        mass_state = self.mass.dxdt(state[2:], time, muscle_force)

        return np.concatenate((muscle_state, mass_state), axis=0)

    def step(self, x0, time, stimulation, muscle_length=None):
        """ Step the system."""
        args = (stimulation, muscle_length)
        res = integrate(self.muscle_mass_system, x0, time, args=args)
        return res

