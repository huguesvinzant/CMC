"""CMC robot"""

import os
import numpy as np
from network import SalamanderNetwork


class SalamanderCMC(object):
    """Salamander robot for CMC"""

    N_BODY_JOINTS = 10
    N_LEGS = 4

    def __init__(self, robot, n_iterations, **kwargs):
        super(SalamanderCMC, self).__init__()
        self.robot = robot
        timestep = int(robot.getBasicTimeStep())
        freqs = kwargs.pop("freqs", None)
        amplitudes = kwargs.pop("amplitudes", None)
        phase_lag = kwargs.pop("phase_lag", None)
        turn = kwargs.pop("turn", None)
        self.network = SalamanderNetwork(
            1e-3*timestep,
            freqs,
            amplitudes,
            phase_lag,
            turn
        )

        # Position sensors
        self.position_sensors = [
            self.robot.getPositionSensor('position_sensor_{}'.format(i+1))
            for i in range(self.N_BODY_JOINTS)
        ]
        for sensor in self.position_sensors:
            sensor.enable(timestep)

        # GPS
        self.gps = robot.getGPS("fgirdle_gps")
        self.gps.enable(timestep)

        # Get motors
        self.motors_body = [
            self.robot.getMotor("motor_{}".format(i+1))
            for i in range(self.N_BODY_JOINTS)
        ]
        self.motors_legs = [
            self.robot.getMotor("motor_leg_{}".format(i+1))
            for i in range(self.N_LEGS)
        ]

        # Set motors
        for motor in self.motors_body:
            motor.setPosition(0)
            motor.enableForceFeedback(timestep)
            motor.enableTorqueFeedback(timestep)
        for motor in self.motors_legs:
            motor.setPosition(-np.pi/2)

        # Iteration counter
        self.iteration = 0

        # Logging
        self.log = ExperimentLogger(
            n_iterations,
            n_links=1,
            n_joints=self.N_BODY_JOINTS,
            filename=kwargs.pop("logs", "logs/log.npz"),
            timestep=1e-3*timestep,
            freqs=freqs,
            amplitude=amplitudes,
            phase_lag=phase_lag,
            turn=turn
        )

    def log_iteration(self):
        """Log state"""
        self.log.log_link_positions(self.iteration, 0, self.gps.getValues())
        for i, motor in enumerate(self.motors_body):
            # Position
            self.log.log_joint_position(
                self.iteration, i,
                self.position_sensors[i].getValue()
            )
            # Velocity
            self.log.log_joint_velocity(
                self.iteration, i,
                motor.getVelocity()
            )
            # Command
            self.log.log_joint_cmd(
                self.iteration, i,
                motor.getTargetPosition()
            )
            # Torque
            self.log.log_joint_torque(
                self.iteration, i,
                motor.getTorqueFeedback()
            )
            # Torque feedback
            self.log.log_joint_torque_feedback(
                self.iteration, i,
                motor.getTorqueFeedback()
            )

    def step(self):
        """Step"""
        # Increment iteration
        self.iteration += 1

        # Update network
        self.network.step()
        positions = self.network.get_motor_position_output()

        # Update control
        for i in range(self.N_BODY_JOINTS):
            self.motors_body[i].setPosition(positions[i])

        # Log data
        self.log_iteration()


class ExperimentLogger(object):
    """Experiment logger"""

    ID_J = {
        "position": 0,
        "velocity": 1,
        "cmd": 2,
        "torque": 3,
        "torque_fb": 4,
        "output": 5
    }
    DTYPE = np.float32

    def __init__(self, n_iterations, n_links, n_joints, filename, **kwargs):
        super(ExperimentLogger, self).__init__()
        # Links: Log position
        self.links = np.zeros([n_iterations, n_links, 3], dtype=self.DTYPE)
        # Joints: Log position, velocity, command, torque, torque_fb, output
        self.joints = np.zeros([n_iterations, n_joints, 6], dtype=self.DTYPE)
        # Network: Log phases, amplitudes, outputs
        self.network = np.zeros(
            [n_iterations, 2*n_joints, 3],
            dtype=self.DTYPE
        )
        # Parameters
        self.parameters = kwargs
        # Filename
        self.filename = filename

    def log_link_positions(self, iteration, link, position):
        """Log link position"""
        self.links[iteration, link, :] = position

    def log_joint_position(self, iteration, joint, position):
        """Log joint position"""
        self.joints[iteration, joint, self.ID_J["position"]] = position

    def log_joint_velocity(self, iteration, joint, velocity):
        """Log joint velocity"""
        self.joints[iteration, joint, self.ID_J["velocity"]] = velocity

    def log_joint_cmd(self, iteration, joint, cmd):
        """Log joint cmd"""
        self.joints[iteration, joint, self.ID_J["cmd"]] = cmd

    def log_joint_torque(self, iteration, joint, torque):
        """Log joint torque"""
        self.joints[iteration, joint, self.ID_J["torque"]] = torque

    def log_joint_torque_feedback(self, iteration, joint, torque_fb):
        """Log joint torque feedback"""
        self.joints[iteration, joint, self.ID_J["torque_fb"]] = torque_fb

    def log_joint_output(self, iteration, joint, output):
        """Log joint output"""
        self.joints[iteration, joint, self.ID_J["output"]] = output

    def save_data(self):
        """Save data to file"""
        # Unlogged initial positions (Step not updated by Webots)
        self.links[0, :, :] = self.links[1, :, :]
        self.joints[0, :, :] = self.joints[1, :, :]
        # Save
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        np.savez(
            self.filename,
            links=self.links,
            joints=self.joints,
            network=self.network,
            **self.parameters
        )

