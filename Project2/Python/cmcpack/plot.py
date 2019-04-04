""" Plotting utils """

import numpy as np
import matplotlib.pyplot as plt
import cmc_pylog as pylog

from .default import DEFAULT


def save_figure(figure, name=None):
    """ Save figure """
    if DEFAULT["save_figures"]:
        for extension in DEFAULT["save_extensions"]:
            fig = figure.replace(" ", "_").replace(".", "dot")
            if name is None:
                name = "{}.{}".format(fig, extension)
            else:
                name = "{}.{}".format(name, extension)
            fig = plt.figure(figure)
            size = plt.rcParams.get('figure.figsize')
            fig.set_size_inches(0.7*size[0], 0.7*size[1], forward=True)
            plt.savefig(name, bbox_inches='tight')
            pylog.debug("Saving figure {}...".format(name))
            fig.set_size_inches(size[0], size[1], forward=True)


def bioplot(data_x, data_y, **kwargs):
    """ Plot data """
    figure = kwargs.pop("figure", "Plot")
    label = kwargs.pop("label", None)
    if isinstance(label, str):
        label = [label]
    linewidth = kwargs.pop("linewidth", 2.0)
    marker = kwargs.pop("marker", "")
    linestyle = kwargs.pop("linestyle", "-")
    n_subs = kwargs.pop("n_subs", 1)
    subs_labels = kwargs.pop("subs_labels", None)
    plt.figure(figure)
    _, axarr = plt.subplots(n_subs, sharex=True, num=figure)
    if n_subs == 1:
        axarr = [axarr]
    n_traces_per_plot = len(np.transpose(data_x))//n_subs
    for i, state in enumerate(np.transpose(data_x)):
        plt.subplot(n_subs, 1, i//n_traces_per_plot+1)
        # axarr[i//n_traces_per_plot]
        plt.plot(
            data_y,
            state,
            label=label[i] if label else "State {}".format(i),
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth
        )
    for i, ax in enumerate(axarr):
        plt.subplot(n_subs, 1, i+1)
        plt.ylabel("State" if not subs_labels else subs_labels[i])
        plt.grid(True)
        ax.set_xlim([min(data_y), max(data_y)])
        leg = plt.legend(loc="best")
        if label is False:
            leg.set_visible(False)
    plt.xlabel("Time [s]")
    save_figure(figure)


def plot_phase(state_list, ode, args, **kwargs):
    """ Plot phase plane """
    figure = kwargs.pop("figure", "Plot")
    label = kwargs.pop("label", [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"])
    scale = kwargs.pop("scale", 0.2)
    n = kwargs.pop("n", 10)
    linewidth = kwargs.pop("linewidth", 2.0)
    plt.figure(figure)
    for s in state_list:
        plot_phase_trajectory(s, linewidth)
    quiver_range = phase_range(state_list)
    plot_phase_quiver(ode, args, quiver_range, scale, n)
    if label is None:
        label = DEFAULT["label"]
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.grid(True)
    save_figure(figure)


def plot_phase_trajectory(state, linewidth):
    """ Plot phase trajectory """
    assert len(state[0, :]) == 2, "State must be 2 dimensions"
    return plt.plot(state[:, 0], state[:, 1], linewidth=linewidth)


def plot_phase_quiver(ode, args, quiver_range, scale=0.2, n=10):
    """ Phase phase quiver """
    X, Y, U, V = phase_compute(ode, quiver_range, scale, n, args)
    plt.quiver(X, Y, U, V, units="xy", angles="xy")


def phase_range(state):
    """ Compute range of phase """
    state = np.array(state)
    # X axis
    min0 = np.min(state[:, :, 0])
    max0 = np.max(state[:, :, 0])
    # Y axis
    min1 = np.min(state[:, :, 1])
    max1 = np.max(state[:, :, 1])
    # Check range and correct of necessary
    if np.abs(max0 - min0) < 1e-6:
        min0, max0 = min0-0.1, max0+0.1
    if np.abs(max1 - min1) < 1e-6:
        min1, max1 = min1-0.1, max1+0.1
    return [[min0, max0], [min1, max1]]


def phase_compute(ode, quiver_range, scale, n, args):
    """ Compute phase """
    min0, max0, min1, max1 = [
        quiver_range[i][j]
        for i in range(2)
        for j in range(2)
    ]
    # Generate grid
    rng = [max0 - min0, max1 - min1]
    X, Y = np.meshgrid(
        np.arange(min0-scale*rng[0], max0+scale*rng[0], rng[0]/float(n)),
        np.arange(min1-scale*rng[1], max1+scale*rng[1], rng[1]/float(n))
    )
    q = np.array([
        [
            ode([X[i, j], Y[i, j]], 0, *args)
            for j, _ in enumerate(X[0, :])
        ] for i, _ in enumerate(X[:, 0])
    ])
    U, V = [
        [
            [
                q[i, j, dim]
                for j, _ in enumerate(X[0, :])
            ] for i, _ in enumerate(X[:, 0])
        ] for dim in range(2)
    ]
    return X, Y, U, V

