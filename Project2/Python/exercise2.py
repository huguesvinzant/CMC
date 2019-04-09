""" Lab 6 Exercise 2

This file implements the pendulum system with two muscles attached

"""

from math import sqrt

import cmc_pylog as pylog
import numpy as np
from matplotlib import pyplot as plt

from cmcpack import DEFAULT
from cmcpack.plot import save_figure
from muscle import Muscle
from muscle_system import MuscleSytem
from neural_system import NeuralSystem
from pendulum_system import PendulumSystem
from system import System
from system_animation import SystemAnimation
from system_parameters import (MuscleParameters, NetworkParameters,
                               PendulumParameters)
from system_simulation import SystemSimulation



# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True
def exercise2b():
    """ Main function to run for Exercise 2b.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """
     # Define and Setup your pendulum model here
    # Check PendulumSystem.py for more details on Pendulum class
    pendulum_params = PendulumParameters()  # Instantiate pendulum parameters
    pendulum_params.L = 0.5  # To change the default length of the pendulum
    pendulum_params.m = 1.  # To change the default mass of the pendulum
    pendulum = PendulumSystem(pendulum_params)  # Instantiate Pendulum object

    #### CHECK OUT PendulumSystem.py to ADD PERTURBATIONS TO THE MODEL #####

    pylog.info('Pendulum model initialized \n {}'.format(
        pendulum.parameters.showParameters()))

    # Define and Setup your pendulum model here
    # Check MuscleSytem.py for more details on MuscleSytem class
    M1_param = MuscleParameters()  # Instantiate Muscle 1 parameters
    M1_param.f_max = 1500  # To change Muscle 1 max force
    M2_param = MuscleParameters()  # Instantiate Muscle 2 parameters
    M2_param.f_max = 1500  # To change Muscle 2 max force
    M1 = Muscle(M1_param)  # Instantiate Muscle 1 object
    M2 = Muscle(M2_param)  # Instantiate Muscle 2 object
    # Use the MuscleSystem Class to define your muscles in the system
    muscles = MuscleSytem(M1, M2)  # Instantiate Muscle System with two muscles
    pylog.info('Muscle system initialized \n {} \n {}'.format(
        M1.parameters.showParameters(),
        M2.parameters.showParameters()))

    # Define Muscle Attachment points
    m1_origin = np.array([-0.17, 0.0])  # Origin of Muscle 1
    m1_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 1

    m2_origin = np.array([0.17, 0.0])  # Origin of Muscle 2
    m2_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 2

    # Attach the muscles
    muscles.attach(np.array([m1_origin, m1_insertion]),
                   np.array([m2_origin, m2_insertion]))

    # Create a system with Pendulum and Muscles using the System Class
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system

    ##### Time #####
    t_max = 20  # Maximum simulation time
    time = np.arange(0., t_max, 0.005)  # Time vector

    ##### Model Initial Conditions #####
    x0_P = np.array([np.pi/4, 0.])  # Pendulum initial condition

    # Muscle Model initial condition
    x0_M = np.array([0., M1.L_OPT, 0., M2.L_OPT])

    x0 = np.concatenate((x0_P, x0_M))  # System initial conditions

    ##### System Simulation #####
    # For more details on System Simulation check SystemSimulation.py
    # SystemSimulation is used to initialize the system and integrate
    # over time

    sim1 = SystemSimulation(sys)  # Instantiate Simulation object

    # Add muscle activations to the simulation
    # Here you can define your muscle activation vectors
    # that are time dependent

    #act1 = np.ones((len(time), 1)) * 1.
    #act2 = np.ones((len(time), 1)) * 0.05
    act1 = np.array([np.sin(time)]).T
    act2 = np.array([-np.sin(time)]).T

    activations = np.hstack((act1, act2))

    # Method to add the muscle activations to the simulation

    sim1.add_muscle_activations(activations)

    # Simulate the system for given time

    sim1.initalize_system(x0, time)  # Initialize the system state

    #: If you would like to perturb the pedulum model then you could do
    # so by
    #sim.sys.pendulum_sys.parameters.PERTURBATION = True
    # The above line sets the state of the pendulum model to zeros between
    # time interval 1.2 < t < 1.25. You can change this and the type of
    # perturbation in
    # pendulum_system.py::pendulum_system function

    # Integrate the system for the above initialized state and time
    sim1.simulate()

    # Obtain the states of the system after integration
    # res is np.array [time, states]
    # states vector is in the same order as x0
    res1 = sim1.results()
    
    
    sim2 = SystemSimulation(sys)  # Instantiate Simulation object
    sim2.add_muscle_activations(activations)

    # Simulate the system for given time

    sim2.initalize_system(x0, time)  # Initialize the system state
    #add perturbation
    sim2.sys.pendulum_sys.parameters.PERTURBATION = True


    # Integrate the system for the above initialized state and time
    sim2.simulate()

    # Obtain the states of the system after integration
    # res is np.array [time, states]
    # states vector is in the same order as x0
    res2 = sim2.results()
    
    # In order to obtain internal states of the muscle
    # you can access the results attribute in the muscle class
    muscle1_results = sim1.sys.muscle_sys.Muscle1.results
    muscle2_results = sim1.sys.muscle_sys.Muscle2.results

    # Plotting the results
    plt.figure('Pendulum')
    plt.title('Pendulum Phase')
    plt.plot(res1[:, 1], res1[:, 2])
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.grid()
    
    plt.figure('Pendulum with perturbation')
    plt.title('Pendulum Phase')
    plt.plot(res2[:, 1], res2[:, 2])
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.grid()
    
    plt.figure('Activation Wave Forms')
    plt.title('Activation Wave Forms')
    plt.plot(time, act1)
    plt.plot(time, act2)
    plt.xlabel('Time [s]')
    plt.ylabel('Activation')
    plt.legend(('Actication muscle 1','Activation muscle 2'))
    plt.grid
    
    poincare_crossings(res1, 0.5, 1, "poincare_cross")

    # To animate the model, use the SystemAnimation class
    # Pass the res(states) and systems you wish to animate
    simulation1 = SystemAnimation(res1, pendulum, muscles)
    simulation2 = SystemAnimation(res2, pendulum, muscles)
    # To start the animation
    if DEFAULT["save_figures"] is False:
        simulation1.animate()
        simulation2.animate()

def exercise2c():
    """ Main function to run for Exercise 2c.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """
     # Define and Setup your pendulum model here
    # Check PendulumSystem.py for more details on Pendulum class
    pendulum_params = PendulumParameters()  # Instantiate pendulum parameters
    pendulum_params.L = 0.5  # To change the default length of the pendulum
    pendulum_params.m = 1.  # To change the default mass of the pendulum
    pendulum = PendulumSystem(pendulum_params)  # Instantiate Pendulum object

    #### CHECK OUT PendulumSystem.py to ADD PERTURBATIONS TO THE MODEL #####

    pylog.info('Pendulum model initialized \n {}'.format(
        pendulum.parameters.showParameters()))

    # Define and Setup your pendulum model here
    # Check MuscleSytem.py for more details on MuscleSytem class
    M1_param = MuscleParameters()  # Instantiate Muscle 1 parameters
    M1_param.f_max = 1500  # To change Muscle 1 max force
    M2_param = MuscleParameters()  # Instantiate Muscle 2 parameters
    M2_param.f_max = 1500  # To change Muscle 2 max force
    M1 = Muscle(M1_param)  # Instantiate Muscle 1 object
    M2 = Muscle(M2_param)  # Instantiate Muscle 2 object
    # Use the MuscleSystem Class to define your muscles in the system
    muscles = MuscleSytem(M1, M2)  # Instantiate Muscle System with two muscles
    pylog.info('Muscle system initialized \n {} \n {}'.format(
        M1.parameters.showParameters(),
        M2.parameters.showParameters()))

    # Define Muscle Attachment points
    m1_origin = np.array([-0.17, 0.0])  # Origin of Muscle 1
    m1_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 1

    m2_origin = np.array([0.17, 0.0])  # Origin of Muscle 2
    m2_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 2

    # Attach the muscles
    muscles.attach(np.array([m1_origin, m1_insertion]),
                   np.array([m2_origin, m2_insertion]))

    # Create a system with Pendulum and Muscles using the System Class
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system

    ##### Time #####
    t_max = 15  # Maximum simulation time
    time = np.arange(0., t_max, 0.005)  # Time vector

    ##### Model Initial Conditions #####
    x0_P = np.array([np.pi/4, 0.])  # Pendulum initial condition

    # Muscle Model initial condition
    x0_M = np.array([0., M1.L_OPT, 0., M2.L_OPT])

    x0 = np.concatenate((x0_P, x0_M))  # System initial conditions

    sim = SystemSimulation(sys)  # Instantiate Simulation object

    plt.figure('Pendulum with different stimulation frequencies')
    frequencies = [0.25,0.5,0.75,1.0,2.0,3.0,4.0,5.0]

    for freq in frequencies:
        act1 = np.array([np.sin(freq*time)]).T
        act2 = np.array([-np.sin(freq*time)]).T
    
        activations = np.hstack((act1, act2))
    
        # Method to add the muscle activations to the simulation
    
        sim.add_muscle_activations(activations)
    
        # Simulate the system for given time
    
        sim.initalize_system(x0, time)  # Initialize the system state
    
    
        # Integrate the system for the above initialized state and time
        sim.simulate()
    
        # Obtain the states of the system after integration
        # res is np.array [time, states]
        # states vector is in the same order as x0
        res = sim.results()
        
    
        # Plotting the results
        plt.plot(res[:, 1], res[:, 2])
        
    plt.title('Pendulum Phase')
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.legend(('0.25','0.5','0.75','1.0','2.0','3.0','4.0','5.0',))
    plt.grid()
    
    plt.figure('Pendulum with different stimulation amplitudes')
    amplitudes = [0.25,0.5,0.75,1.0,2.0,3.0,4.0,5.0]

    for amp in amplitudes:
        act1 = np.array([amp*np.sin(time)]).T
        act2 = np.array([amp*(-np.sin(time))]).T
    
        activations = np.hstack((act1, act2))
    
        # Method to add the muscle activations to the simulation
    
        sim.add_muscle_activations(activations)
    
        # Simulate the system for given time
    
        sim.initalize_system(x0, time)  # Initialize the system state
    
    
        # Integrate the system for the above initialized state and time
        sim.simulate()
    
        # Obtain the states of the system after integration
        # res is np.array [time, states]
        # states vector is in the same order as x0
        res = sim.results()
        
    
        # Plotting the results
        plt.plot(res[:, 1], res[:, 2])
        
    plt.title('Pendulum Phase')
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.legend(('0.25','0.5','0.75','1.0','2.0','3.0','4.0','5.0',))
    plt.grid()
    
    
def poincare_crossings(res, threshold, crossing_index, figure):
    """ Study poincaré crossings """
    ci = crossing_index

    # Extract state of first trajectory
    state = res[:,2]

    # Crossing index (index corrsponds to last point before crossing)
    idx = np.argwhere(np.diff(np.sign(state[ci, 2] - threshold)) < 0)
    # pylog.debug("Indices:\n{}".format(idx))  # Show crossing indices

    # Linear interpolation to find crossing position on threshold
    # Position before crossing
    pos_pre = np.array([state[index[0], 2] for index in idx])
    # Position after crossing
    pos_post = np.array([state[index[0]+1, 2] for index in idx])
    # Position on threshold
    pos_treshold = [
        (
            (threshold - pos_pre[i, 1])/(pos_post[i, 1] - pos_pre[i, 1])
        )*(
            pos_post[i, 0] - pos_pre[i, 0]
        ) + pos_pre[i, 0]
        for i, _ in enumerate(idx)
    ]

    # Plot
    # Figure limit cycle variance
    plt.figure(figure)
    plt.plot(pos_treshold, "o-")
    val_min = np.sort(pos_treshold)[2]
    val_max = np.sort(pos_treshold)[-2]
    bnd = 0.3*(val_max - val_min)
    plt.ylim([val_min-bnd, val_max+bnd])
    plt.xlabel("Number of Poincaré section crossings")
    plt.ylabel(" (Neuron 2 = {})".format(threshold))
    plt.grid(True)

    # Figure limit cycle
    plt.figure(figure+"_phase")
    plt.plot([val_min-0.3, val_max+0.3], [threshold]*2, "gx--")
    for pos in pos_treshold:
        plt.plot(pos, threshold, "ro")

    # Save plots if option activated
    if DEFAULT["save_figures"] is True:
        from cmcpack.plot import save_figure
        save_figure(figure)
        save_figure(figure+"_phase")

        # Zoom on limit cycle
        plt.figure(figure+"_phase")
        plt.xlim([val_min-bnd, val_max+bnd])
        plt.ylim([threshold-1e-7, threshold+1e-7])
        save_figure(figure=figure+"_phase", name=figure+"_phase_zoom")

    return idx


    
def exercise2():
    """ Main function to run for Exercise 2.

    """
    exercise2b();
   
    if not DEFAULT["save_figures"]:
        plt.show()
    else:
        figures = plt.get_figlabels()
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise2()

