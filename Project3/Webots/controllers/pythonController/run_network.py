"""Run network without Webots"""

import time
import numpy as np
import matplotlib.pyplot as plt
from network import SalamanderNetwork
from save_figures import save_figures
from parse_args import save_plots


def main(plot=True):
    """Main - Run network without Webots and plot results"""
    # Simulation setup
    timestep = 1e-3
    times = np.arange(0, 2, timestep)
    freqs = 1
    amplitudes = None
    phase_lag = None
    turn = None
    amplitudes = [1, 1]
    phase_lag = 2*np.pi/(10-1)
    turn = 0
    network = SalamanderNetwork(timestep, freqs, amplitudes, phase_lag, turn)

    # Logs
    phases_log = np.zeros([
        len(times),
        len(network.phase_equation.phases)
    ])
    phases_log[0, :] = network.phase_equation.phases
    amplitudes_log = np.zeros([
        len(times),
        len(network.amplitude_equation.amplitudes)
    ])
    amplitudes_log[0, :] = network.amplitude_equation.amplitudes
    outputs_log = np.zeros([
        len(times),
        len(network.get_motor_position_output())
    ])
    outputs_log[0, :] = network.get_motor_position_output()

    # Simulation
    tic = time.time()
    for i, _ in enumerate(times[1:]):
        network.step()
        phases_log[i+1, :] = network.phase_equation.phases
        amplitudes_log[i+1, :] = network.amplitude_equation.amplitudes
        outputs_log[i+1, :] = network.get_motor_position_output()
    toc = time.time()

    # Simulation information
    print("Time to run simulation for {} steps: {} [s]".format(
        len(times),
        toc - tic
    ))

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

