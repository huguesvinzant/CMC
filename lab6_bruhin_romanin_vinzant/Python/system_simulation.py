import numpy as np
from cmcpack import integrate
import pdb
import sys


class SystemSimulation(object):
    """System Simulation

    """

    def __init__(self, sys):
        super(SystemSimulation, self).__init__()
        self.sys = sys
        self.muscle_activations = None
        self.ext_in = None

    def add_muscle_activations(self, act):
        """Function applies the array of muscle activations during time
        integration Keyword Arguments: -- act <2D array>
        Parameters
        ----------
        act: <array>
            2D array of activation values for each time instant

        """
        self.muscle_activations = act

    def add_external_inputs_to_network(self, ext_in=None):
        """Function to add the external inputs to the neural network

        Parameters
        ----------
        ext_in: np.ndarray
            External inputs to each neuron in the network.
            Range of the inputs is [0, 1]
            The array is np.ndarray containing external inputs to
            each neuron at time t
        """
        self.ext_in = ext_in

    def _get_current_external_input_to_network(self, time):
        """Function to get the current external input to network.

        Parameters
        ----------
        self: type
            description
        time: float
            Current simulation time

        Returns
        -------
        current_ext_in : np.array
            Current external input to each neuron at time t
        """

        if self.ext_in is not None:
            index = np.argmin((self.time - time)**2)
            return np.array(self.ext_in[index, :])

        return np.zeros(4)

    def _get_current_muscle_activation(self, time, state):
        """Function to return the current muscle activation to be applied
        during integration.

        """
        if self.sys.systems_list.count('neural') == 1:
            # Apply the activation function to the neuron state m
            # [WIP]
            neural_act = self.sys.neural_sys.n_act(state[6:])
            return np.array([neural_act[0], neural_act[1]])
        else:
            if self.muscle_activations is not None:
                index = np.argmin((self.time - time)**2)
                return np.array(self.muscle_activations[index, :])
            else:
                return np.array([0.05, 0.05])

    def muscle_joint_interface(self, time, state):
        torque = self.sys.muscle_sys.compute_muscle_torque(
            state)
        return torque

    def initalize_system(self, x0, time, *args):
        """Initialize the system to start simulation.

        Parameters
        ----------
        x0: numpy.array
            Initial states of the models on the system
        time: numpy.array
            Time vector for the system to be integrated for
        args: tuple
            external args for the integrator

        """

        self.x0 = x0
        self.time = time
        self.args = args

        # Initialize muscle states
        init_muscle_lce = self.sys.muscle_sys.initialize_muscle_length(
            self.x0[0])

        self.x0[3] = init_muscle_lce[0]
        self.x0[5] = init_muscle_lce[1]

        # Validate the muscle attachment points
        valid = self.sys.muscle_sys.validate_muscle_attachment(
            self.sys.pendulum_sys.parameters)
        if(not valid):
            sys.exit(1)

    def derivative(self, state, time, *args):
        #: Unwrap the args
        angle = state[0]

        # Compute the joint torques from muscle forces
        torque = self.muscle_joint_interface(time, state)

        muscle_lengths = self.sys.muscle_sys.length_from_angle(angle)
        muscle_activations = self._get_current_muscle_activation(time, state)

        m_der = self.sys.muscle_sys.derivative(
            state[2:6], time, muscle_activations, muscle_lengths)

        p_der = self.sys.pendulum_sys.pendulum_system(time,
                                                      state[0],
                                                      state[1], torque)

        if (self.sys.systems_list.count('neural') == 1.0):
            self.sys.neural_sys.external_inputs(
                self._get_current_external_input_to_network(time))
            n_der = self.sys.neural_sys.derivative(
                time, state[6:])
            update = np.concatenate((p_der, m_der, n_der), axis=0)
        else:
            update = np.concatenate((p_der, m_der), axis=0)
        return update

    def simulate(self):
        #: Run the integrator fpr specified time
        self.res = integrate(self.derivative, self.x0, self.time,
                             args=self.args, rk=True, tol=True)

    def results(self):
        """Return the state of the system after integration.
        The function adds the time vector to the integrated system states."""

        #: Instatiate the muscle results container
        self.sys.muscle_sys.Muscle1.instantiate_result_from_state(
            self.time)
        self.sys.muscle_sys.Muscle2.instantiate_result_from_state(
            self.time)

        angle = self.res[:, 1]
        muscle1_state = self.res[:, 2:4]
        muscle2_state = self.res[:, 4:6]

        Muscle1 = self.sys.muscle_sys.Muscle1
        Muscle2 = self.sys.muscle_sys.Muscle2

        for idx, _time in enumerate(self.time):
            #: Compute muscle lengths from angle
            muscle_lengths = self.sys.muscle_sys.length_from_angle(
                angle[idx])
            Muscle1.generate_result_from_state(idx, _time,
                                               muscle_lengths[0],
                                               muscle1_state[idx][:])
            Muscle2.generate_result_from_state(idx, _time,
                                               muscle_lengths[1],
                                               muscle2_state[idx][:])

        return np.concatenate(
            (np.expand_dims(self.time, axis=1), self.res), axis=1)

