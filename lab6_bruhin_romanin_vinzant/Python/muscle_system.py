import numpy as np
import math
import cmc_pylog as pylog


class MuscleSytem(object):
    """Class comprising of the antagonist muscle pair
    """

    def __init__(self, Muscle1, Muscle2):
        super(MuscleSytem, self).__init__()
        self.Muscle1 = Muscle1
        self.Muscle2 = Muscle2

        (self.muscle1_length,
         self.muscle2_length) = self.compute_default_muscle_length()

        #: Attachment Point Attributes
        self.a1_m1 = 0.0
        self.a2_m1 = 0.0
        self.a1_m2 = 0.0
        self.a2_m2 = 0.0

        # Muscle 1 acts in negative torque direction
        self.dir1 = -1.0
        # Muscle 2 acts in positive torque direction
        self.dir2 = 1.0

        #: Muscle Position
        self.muscle1_pos = 0.0
        self.muscle2_pos = 0.0

    def attach(self, muscle1_pos, muscle2_pos):
        """ Muscle attachment points.

        Parameters:
        -----------
            muscle1_pos : <numpy.array>
                Attachment point of muscle 1.
                [origin,
                 insertion]
            muscle2_pos : <numpy.array>
                Attachment point of muscle 2.
                [origin,
                 insertion]

        Example:
        --------
        >>> muscle1_pos = numpy.array([[-5.0, 0.0],
                                       [0.0, 1.0]])
        >>> muscle2_pos = numpy.array([[5.0, 0.0],
                                       [0.0, 1.0]])
        """
        self.muscle1_pos = muscle1_pos
        self.muscle2_pos = muscle2_pos

        self.compute_attachment_distances()

    def compute_attachment_distances(self):
        """ Compute the distances between the pendulum joint
            and muscle attachment origin and inertion point.
        """

        # Muscle 1
        self.a1_m1 = np.linalg.norm(
            self.muscle1_pos[0] - np.array([0, 0]))
        self.a2_m1 = np.linalg.norm(
            self.muscle1_pos[1] - np.array([0, 0]))

        # Muscle 2
        self.a1_m2 = np.linalg.norm(
            self.muscle2_pos[0] - np.array([0, 0]))
        self.a2_m2 = np.linalg.norm(
            self.muscle2_pos[1] - np.array([0, 0]))

    def compute_default_muscle_length(self):
        """ Compute the default length of the muscles. """

        m1_l_mtc = self.Muscle1.L_SLACK + self.Muscle1.L_OPT
        m2_l_mtc = self.Muscle2.L_SLACK + self.Muscle2.L_OPT

        return m1_l_mtc, m2_l_mtc

    def initialize_muscle_length(self, angle):
        """Initialize the muscle contractile and tendon length.

        Parameters
        ----------
        self: type
            description
        angle: float
            Initial position of the pendulum [rad]

        """

        muscle_length = self.length_from_angle(angle)

        l_ce_1 = self.Muscle1.initialize_muscle_length(muscle_length[0])
        l_ce_2 = self.Muscle2.initialize_muscle_length(muscle_length[1])
        return np.array([l_ce_1, l_ce_2])

    def validate_muscle_attachment(self, parameters):
        """Validate the muscle attachment positions.

        Provided pendulum length and muscle attachments
        check if the muscle attachments are valid or not!

        Parameters
        ----------
        parameters: PendulumParameters
            Pendulum parameters

        Returns
        -------
        check : bool
            Returns if the muscle attachments are valid or not
        """

        check = (parameters.L > abs(self.muscle1_pos[1, 1])) and (
            parameters.L > abs(self.muscle2_pos[1, 1])) and (
                self.muscle1_pos[1, 0] == 0.0) and (
                    self.muscle2_pos[1, 0] == 0.0)

        if check:
            pylog.info('Validated muscle attachments')
            return check
        pylog.error('Invalid muscle attachment points')
        return check

    def position_from_angle(self, angle):
        """ Compute the muscle position from joint angle.

        Parameters:
        -----------
            angle : <float>
                Pendulum angle

        Returns:
        --------
            muscle1_pos : <float>
                Updates Attachment points of muscle 1
            muscle2_pos : <float>
                Updates Attachment points of muscle 2
        """

        def rot_matrix(x): return np.array(
            [[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])

        # Update muscle attachment points on the pendulum
        pos1 = np.array([self.muscle1_pos[0, :], np.dot(
            rot_matrix(angle), self.muscle1_pos[1, :])])

        pos2 = np.array([self.muscle2_pos[0, :], np.dot(
            rot_matrix(angle), self.muscle2_pos[1, :])])

        return np.concatenate(([pos1], [pos2]))

    def length_from_angle(self, angle):
        """ Compute the muscle length from joint angle.

        Parameters:
        -----------
            angle : <float>
                Pendulum angle

        Returns:
        --------
            muscle1_length : <float>
                Muscle 1 length
            muscle2_length : <float>
                Muscle 2 length
        """

        _pos = self.position_from_angle(angle)

        self.muscle1_length = np.linalg.norm(
            _pos[0][0] - _pos[0][1])

        self.muscle2_length = np.linalg.norm(
            _pos[1][0] - _pos[1][1])

        return np.array([self.muscle1_length,
                         self.muscle2_length])

    def compute_moment_arm(self, angle):
        """ Compute the moment arm of the muscles based on the joint angle.

        moment = a1*a2

        Parameters
        ----------
        angle: float
            Current angle of the pendulum

        """
        # Muscle 1 moment arm
        self.moment1 = self.a1_m1 * self.a2_m1 * \
            np.cos(angle) / self.muscle1_length
        # Muscle 2 moment arm
        self.moment2 = self.a1_m2 * self.a2_m2 * \
            np.cos(angle) / self.muscle2_length

    def compute_muscle_torque(self, state):
        """ Keyword Arguments:
            angle -- Angle of the Pendulum """

        angle = state[0]

        _pos = self.length_from_angle(angle)

        #: Compute tendon slack length from current state
        l_se_1 = _pos[0] - state[3]
        l_se_2 = _pos[1] - state[5]

        self.compute_moment_arm(angle)
        muscle1_torque = self.dir1 * self.moment1 * \
            self.Muscle1.compute_tendon_force(l_se_1)
        muscle2_torque = self.dir2 * self.moment2 * \
            self.Muscle2.compute_tendon_force(l_se_2)
        return muscle1_torque + muscle2_torque

    def derivative(self, state, time, *args):
        """ Keyword Arguments:
            self  --
            state --
            time  --
            *args --  """

        # Set the change in muscle length

        stimulations = args[0]
        lengths = args[1]

        # Update and retrieve the derivatives
        return np.concatenate(
            (self.Muscle1.dxdt(
                state[: 2], time, stimulations[0], lengths[0]),
             self.Muscle2.dxdt(
                 state[2:], time, stimulations[1], lengths[1])))

