"""Python controller"""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, LED, DistanceSensor
import numpy as np
import cmc_pylog as pylog
from controller import Supervisor
from cmc_robot import SalamanderCMC


class RobotResetControl(object):
    """Robot reset control"""

    def __init__(self, world, n_joints):
        super(RobotResetControl, self).__init__()
        self.world = world
        self.n_joints = n_joints
        self.salamander = self.world.getFromDef("SALAMANDER")
        self.initial_position = np.array(
            self.salamander.getField("translation").getSFVec3f()
        )
        self.initial_rotation = np.array(
            self.salamander.getField("rotation").getSFRotation()
        )
        self.solid_links = [
            self.world.getFromDef("SOLID_{}".format(i+1))
            for i in range(10)
        ]
        self.hinge_joints = [
            self.world.getFromDef("JOINT_PARAM_{}".format(i+1))
            for i in range(10)
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
        for i in range(self.n_joints):
            self.hinge_joints[i].getField("position").setSFFloat(0)
            # self.hinge_joints[i].setVelocity([0, 0, 0, 0, 0, 0])
            self.solid_links[i].setVelocity([0, 0, 0, 0, 0, 0])


def run_simulation(world, parameters, timestep, n_iterations, logs):
    """Run simulation"""

    # Set parameters
    pylog.info((
        "Running new simulation:"
        "\n  - Amplitude: {}"
        "\n  - Phase lag: {}"
        "\n  - Turn: {}"
    ).format(parameters[0], parameters[1], parameters[2]))

    # Setup salamander control
    salamander = SalamanderCMC(
        world,
        n_iterations,
        logs=logs,
        freqs=parameters[0],
        amplitudes=parameters[1],
        phase_lag=parameters[2],
        turn=parameters[3]
    )

    # Simulation
    iteration = 0
    while world.step(timestep) != -1:
        iteration += 1
        if iteration >= n_iterations:
            break
        salamander.step()

    # Log data
    pylog.info("Logging simulation data to {}".format(logs))
    salamander.log.save_data()


def main():
    """Main"""

    # Duration of each simulation
    simulation_duration = 20

    # Get supervisor to take over the world
    world = Supervisor()
    n_joints = 10
    timestep = int(world.getBasicTimeStep())
    freqs = 1

    # Get and control initial state of salamander
    reset = RobotResetControl(world, n_joints)

    # Simulation example
    amplitude = None
    phase_lag = None
    turn = None
    parameter_set = [
        [freqs, amplitude, phase_lag, turn],
        [freqs, amplitude, phase_lag, turn]
    ]
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*simulation_duration/timestep),
            logs="./logs/example/simulation_{}.npz".format(simulation_i)
        )

    # Pause
    world.simulationSetMode(world.SIMULATION_MODE_PAUSE)
    pylog.info("Simulations complete")
    world.simulationQuit(0)


if __name__ == '__main__':
    main()

