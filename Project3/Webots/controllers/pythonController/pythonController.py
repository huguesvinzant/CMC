"""Python controller"""

import cmc_pylog as pylog
from controller import Supervisor
from reset import RobotResetControl
from exercise_example import exercise_example
from exercise_9b import exercise_9b
from exercise_9c import exercise_9c
from exercise_9d import exercise_9d1, exercise_9d2
from exercise_9f import exercise_9f
from exercise_9g import exercise_9g


def main():
    """Main"""

    # Get supervisor to take over the world
    world = Supervisor()
    n_joints = 10
    timestep = int(world.getBasicTimeStep())

    # Get and control initial state of salamander
    reset = RobotResetControl(world, n_joints)

    # Simulation arguments
    arguments = world.getControllerArguments()
    pylog.info("Arguments passed to smulation: {}".format(arguments))

    # Exercise example to show how to run a grid search
    if "example" in arguments:
        exercise_example(world, timestep, reset)

    # Exercise 9b - Phase lag + amplitude study
    if "9b" in arguments:
        exercise_9b(world, timestep, reset)

    # Exercise 9c - Gradient amplitude study
    if "9c" in arguments:
        exercise_9c(world, timestep, reset)

    # Exercise 9d1 - Turning
    if "9d1" in arguments:
        exercise_9d1(world, timestep, reset)

    # Exercise 9d2 - Backwards swimming
    if "9d2" in arguments:
        exercise_9d2(world, timestep, reset)

    # Exercise 9f - Walking
    if "9f" in arguments:
        exercise_9f(world, timestep, reset)

    # Exercise 9g - Transitions
    if "9g" in arguments:
        exercise_9g(world, timestep, reset)

    # Pause
    world.simulationSetMode(world.SIMULATION_MODE_PAUSE)
    pylog.info("Simulations complete")
    world.simulationQuit(0)


if __name__ == '__main__':
    main()

