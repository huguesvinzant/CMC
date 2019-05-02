"""Experiment logger"""

import os
import numpy as np


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

