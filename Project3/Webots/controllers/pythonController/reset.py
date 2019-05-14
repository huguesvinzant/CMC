"""Robot reset during experiments"""

import numpy as np


class RobotResetControl(object):
    """Robot reset control"""

    def __init__(self, world, n_joints):
        super(RobotResetControl, self).__init__()
        self.world = world
        self.n_joints_body = 10
        self.n_joints_legs = 4
        self.salamander = self.world.getFromDef("SALAMANDER")
        self.initial_position = np.array(
            self.salamander.getField("translation").getSFVec3f()
        )
        self.initial_rotation = np.array(
            self.salamander.getField("rotation").getSFRotation()
        )
        self.solid_links = [
            self.world.getFromDef("SOLID_{}".format(i+1))
            for i in range(self.n_joints_body)
        ]
        self.hinge_joints = [
            self.world.getFromDef("JOINT_PARAM_{}".format(i+1))
            for i in range(self.n_joints_body)
        ] + [
            self.world.getFromDef("JOINT_PARAM_LEG_{}".format(i+1))
            for i in range(self.n_joints_legs)
        ]

    def reset(self):
        """Reset state"""
        self.reset_pose()
        self.reset_internal()

    def reset_pose(self):
        """Reset robot pose"""
        self.salamander.getField("translation").setSFVec3f(
            self.initial_position.tolist()
        )
        self.salamander.getField("rotation").setSFRotation(
            self.initial_rotation.tolist()
        )

    def reset_internal(self):
        """Reset intenal links and joints states"""
        for i in range(self.n_joints_body+self.n_joints_legs):
            self.hinge_joints[i].getField("position").setSFFloat(0)
            # self.hinge_joints[i].setVelocity([0, 0, 0, 0, 0, 0])
        for i in range(self.n_joints_body):
            self.solid_links[i].setVelocity([0, 0, 0, 0, 0, 0])

