""" Lab 5 - Exercise 1 """

import matplotlib.pyplot as plt
import numpy as np

import cmc_pylog as pylog
from muscle import Muscle
from mass import Mass
from cmcpack import DEFAULT, parse_args
from cmcpack.plot import save_figure
from system_parameters import MuscleParameters, MassParameters
from isometric_muscle_system import IsometricMuscleSystem
from isotonic_muscle_system import IsotonicMuscleSystem

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True


def exercise1a():
    """ Exercise 1a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Defination of muscles
    parameters = MuscleParameters()
    pylog.warning("Loading default muscle parameters")
    pylog.info(parameters.showParameters())
    pylog.info("Use the parameters object to change the muscle parameters")

    # Create muscle object
    muscle = Muscle(parameters)

    # Instatiate isometric muscle system
    sys = IsometricMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)
    
    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT]
    # x0[0] --> muscle stimulation intial value
    # x0[1] --> muscle contracticle length initial value
    
    # Set the time for integration
    t_start = 0.0
    t_stop = 0.5
    time_step = 0.001
    time = np.arange(t_start, t_stop, time_step)
    
    # Evalute for a single muscle stimulation
    muscle_stimulation = np.arange(0,1,0.2)
    
    # Several muscle stretch
    muscle_stretches = np.arange(0,0.3,0.01)
    
    active_active = []

    for stim in muscle_stimulation:
        active_forces = []
        passive_forces = []
        total = [] 
        lengths = []
        for stretch in muscle_stretches:
            # Run the integration
            result = sys.integrate(x0=x0,
                                   time=time,
                                   time_step=time_step,
                                   stimulation=stim,
                                   muscle_length=stretch)
            active_forces.append(result.active_force[-1])
            passive_forces.append(result.passive_force[-1])
            total.append(result.active_force[-1]+result.passive_force[-1])
            lengths.append(result.l_ce[-1])
        active_active.append(active_forces)
    
    # Plotting
    plt.figure('Isometric muscle experiment 1')
    plt.plot(lengths, active_forces)
    plt.plot(lengths, passive_forces)
    plt.plot(lengths, total)
    plt.title('Isometric muscle experiment stimulation')
    plt.xlabel('Muscle stretch')
    plt.ylabel('Muscle force')
    plt.legend(('Active','Passive','Total'))
    plt.grid()
    plt.show()
    
    # Plotting
    plt.figure('Isometric muscle experiment 2')
    for i in range(len(muscle_stimulation)):
        plt.plot(lengths, active_active[i])
    plt.title('Isometric muscle experiment')
    plt.xlabel('Muscle stretch')
    plt.ylabel('Muscle force')
    plt.legend(muscle_stimulation)
    plt.grid()
    plt.show()
    
    # Plotting
    #plt.figure('Isotonic muscle experiment')
    #plt.plot(result.time, result.v_ce)
    #plt.title('Isotonic muscle experiment')
    #plt.xlabel('Time [s]')
    #plt.ylabel('Muscle contractilve velocity')
    #plt.grid()
    
    #muscle with longer l_opt
    muscle.L_OPT = 0.5
    muscle_stimulation = 1.
    lce = []
    totalF = []
    activeF=[]
    passiveF=[]
    for stretch in muscle_stretches:
        # Run the integration
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               stimulation=muscle_stimulation,
                               muscle_length=stretch)
        activeF.append(result.active_force[-1])
        passiveF.append(result.passive_force[-1])
        lce.append(result.l_ce[-1])
        totalF.append(result.active_force[-1]+result.passive_force[-1])
    plt.figure('muscle with l_opt=0.5')   
    plt.title('muscle with l_opt=0.5')
    plt.plot(lce, activeF)
    plt.plot(lce, passiveF)
    plt.plot(lce, totalF)
    plt.xlabel('Muscle Stretch')
    plt.ylabel('Force')
    plt.ylim((0,4000))
    plt.legend(('Active Force','Passive Force','Total Force'))

    plt.grid()
    
    
    
    #muscle with shorter l_opt
    t_start = 0.0
    t_stop = 1
    time_step = 0.005

    time = np.arange(t_start, t_stop, time_step)
    muscle_stretches = np.arange(0,0.3,0.01 )
    muscle.L_OPT = 0.075
    muscle_stimulation = 1.
    lce = []
    totalF = []
    activeF=[]
    passiveF=[]
    plt.figure('muscle with l_opt=0.075')   

    for stretch in muscle_stretches:
        # Run the integration
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               stimulation=muscle_stimulation,
                               muscle_length=stretch)
        activeF.append(result.active_force[-1])
        passiveF.append(result.passive_force[-1])
        lce.append(result.l_ce[-1])
        totalF.append(result.active_force[-1]+result.passive_force[-1])
    plt.title('muscle with l_opt=0.075')
    plt.plot(lce, activeF)
    plt.plot(lce, passiveF)
    plt.plot(lce, totalF)
    plt.xlabel('Muscle Stretch')
    plt.ylabel('Force')
    plt.ylim((0,4000))
    plt.legend(('Active Force','Passive Force','Total Force'))
    plt.grid()


def exercise1d():
    """ Exercise 1d

    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed contract. The instantaneous velocity at which the muscle
    contracts is of our interest."""

    # Defination of muscles
    muscle_parameters = MuscleParameters()
    print(muscle_parameters.showParameters())

    mass_parameters = MassParameters()
    print(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)

    # Create mass object
    mass = Mass(mass_parameters)

    pylog.warning("Isotonic muscle contraction to be implemented")

    # Instatiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length

    # Evalute for a single load
    load = 100.

    # Evalute for a single muscle stimulation
    muscle_stimulation = 1.

    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT,
          sys.muscle.L_OPT + sys.muscle.L_SLACK, 0.0]
    
    # x0[0] - -> activation
    # x0[1] - -> contractile length(l_ce)
    # x0[2] - -> position of the mass/load
    # x0[3] - -> velocity of the mass/load
    

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.3
    time_step = 0.001
    time_stabilize = 0.2

    time = np.arange(t_start, t_stop, time_step)
    
    loads = np.arange(1, 500, 10)
    
    velocities = []

    for index, load in enumerate(loads):
        
        # Run the integration
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               time_stabilize=time_stabilize,
                               stimulation=muscle_stimulation,
                               load=load)                

        if (result.l_mtc[0] < sys.muscle.L_OPT + sys.muscle.L_SLACK):
            velocities.append(np.max(result.v_ce))
            print('max')
        else:
            velocities.append(np.min(result.v_ce))
            print('min')


    #Muscle contracile Velocity - Tension (load) relationship
    
    plt.figure('Isotonic muscle experiment')
    plt.xlabel('Muscle Contractile Velocity [m/s]')
    plt.ylabel('Tension [N]')
    plt.plot(velocities, loads)
    
    # Plotting
    #plt.figure('Isometric muscle experiment')
    #plt.plot(result.time, result.tendon_force)
    #plt.title('Isometric muscle experiment')
    #plt.xlabel('Time [s]')
    #plt.ylabel('Muscle Force')
    #plt.grid()


def exercise1():
    #exercise1a()
    exercise1d()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        print(figures)
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise1()

