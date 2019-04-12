""" Results handler """

import numpy as np
from .plot import (
    bioplot,
    plot_phase
)


class ResultData(object):
    """ ODE result data

    >>> ResultData([0, 1, 2], [0, 1, 2])
    State:
    [[0], [1], [2]]
    Time:
    [0, 1, 2]
    ODE:
    None
    Args:
    ()
    """

    def __init__(self, state, time, ode=None, args=()):
        super(ResultData, self).__init__()
        if len(np.shape(state)) == 1:
            state = [[s] for s in state]
        self._state = state
        self._time = time
        self._ode = ode
        self._args = args

    @property
    def state(self):
        """ State after integration """
        return self._state

    @property
    def time(self):
        """ State after integration """
        return self._time

    @property
    def ode(self):
        """ ODE used to obtain this result """
        return self._ode

    @property
    def args(self):
        """ Arguments given to ODE before obtaining this result """
        return self._args

    def __repr__(self):
        return self._msg()

    def __str__(self):
        return self._msg()

    def _msg(self):
        """ Message for printing """
        return "State:\n{}\nTime:\n{}\nODE:\n{}\nArgs:\n{}".format(
            self.state, self.time, self.ode, self.args
        )


class Result(ResultData):
    """ A class dedicated to containing and visualising ODE integration results

    >>> Result(state=[0, 1, 2], time=[0, 1, 2])
    State:
    [[0], [1], [2]]
    Time:
    [0, 1, 2]
    ODE:
    None
    Args:
    ()

    """

    def __init__(self, state, time, ode=None, args=()):
        super(Result, self).__init__(state, time, ode, args)

    def plot_state(self, figure="Plot", label=None, **kwargs):
        """ Plot results """
        return bioplot(
            self.state,
            self.time,
            figure=figure,
            label=label,
            marker=kwargs.pop("marker", ""),
            linestyle=kwargs.pop("linestyle", "-"),
            linewidth=kwargs.pop("linewidth", 2.0),
            n_subs=kwargs.pop("n_subs", 1)
        )

    def plot_phase(self, figure=None, label=None, **kwargs):
        """ Plot phase plane """
        if self.ode is None:
            msg = "Cannot plot phase, ODE was not defined in {}"
            raise Exception(msg.format(self.__class__))
        return plot_phase(
            [self.state], self.ode, self.args,
            figure=figure,
            label=label,
            scale=kwargs.pop("scale", 0.2),
            n=kwargs.pop("n", 15),
            linewidth=kwargs.pop("linewidth", 2.0)
        )

    def plot_angle(self, figure=None, label=None, **kwargs):
        """ Plot state angle """
        number_traces = (len(np.transpose(self.state)))//2
        state_angle = np.transpose([
            np.arctan2(self.state[:, 2*i+1], self.state[:, 2*i])
            for i in range(number_traces)
        ])
        return bioplot(
            state_angle,
            self.time,
            figure=figure,
            label=label,
            marker=kwargs.pop("marker", ""),
            linestyle=kwargs.pop("linestyle", "-"),
            linewidth=kwargs.pop("linewidth", 2.0),
            n_subs=kwargs.pop("n_subs", 1)
        )


class MultipleResultsODE(ResultData):
    """ Multiple Results ODE """

    def __init__(self, state, time, ode=None, args=()):
        super(MultipleResultsODE, self).__init__(state, time, ode, args)

    def plot_state(self, figure=None, label=None, **kwargs):
        """ plot phase plane """
        scale = kwargs.pop("scale", 0.2)
        linewidth = kwargs.pop("linewidth", 2.0)
        n_subs = kwargs.pop("n_subs", 2)
        subs_labels = kwargs.pop("subs_labels", None)
        return [
            bioplot(
                state, self.time,
                figure=figure,
                label=label,
                scale=scale,
                linewidth=linewidth,
                n_subs=n_subs,
                subs_labels=subs_labels
            )
            for state in self.state
        ]

    def plot_phase(self, figure=None, label=None, **kwargs):
        """ plot phase plane """
        return plot_phase(
            self.state, self.ode, self.args,
            figure=figure,
            label=label,
            scale=kwargs.pop("scale", 0.2),
            n=kwargs.pop("n", 15),
            linewidth=kwargs.pop("linewidth", 2.0)
        )


if __name__ == '__main__':
    import doctest
    doctest.testmod()

