""" ODE integration """
from __future__ import print_function
from scipy.integrate import odeint, ode
import numpy as np
from .results import (
    Result,
    MultipleResultsODE
)


def integrate(ode_fun, x0, time, args=(), rk=False, tol=False):
    """ Integrate ode """
    if not (rk and tol):
        result = integrate_lsoda(ode_fun, x0, time, args=args)
    elif (rk and tol):
        result = integrate_rk_tol(ode_fun, x0, time, args=args)
    elif (rk and not tol):
        result = integrate_rk(ode_fun, x0, time, args=args)
    return result


def integrate_lsoda(ode_fun, x0, time, args=(), full_output=False):
    """ Integrate ode """
    x, out = odeint(ode_fun, x0, time, args=tuple(
        args), full_output=True, mxstep=100)
    r = Result(x, time, ode_fun, args)
    return r if full_output is False else (r, out)


def integrate_rk(ode_fun, x0, time, args=()):
    """ Integrate ode """
    def ode_inv(t, y, _, *f_args):
        """ Function wrapper to invert t and y for RK integration """
        return ode_fun(y, t, *f_args)
    r = ode(ode_inv)
    r.set_integrator("dopri5", method="bdf", nsteps=100)
    r.set_initial_value(y=x0, t=time[0])
    r.set_f_params(tuple(args))
    x = np.array([r.integrate(t) for t in time])
    return Result(x, time, ode_fun, args)


def integrate_rk_tol(ode_fun, x0, time, args=()):
    """ Integrate ode.

    This function handles numerical instabilities by changing the integration
    tolerance values."""
    def ode_inv(t, y, _,  *f_args):
        """ Function wrapper to invert t and y for RK integration """
        return ode_fun(y, t, *args)
    r = ode(ode_inv)
    r.set_integrator("dopri5", method="bdf",
                     atol=1e-4, rtol=1e-4)
    r.set_initial_value(y=x0, t=time[0])
    r.set_f_params(tuple(args))

    def fun(r, t, i, time_len):
        r.integrate(t)
        if(i % int(time_len / 10) == 0):
            print('Integration at {}%'.format(int(i / float(time_len) * 100)))
        return r.y
    time_len = len(time)
    x = np.array([fun(r, t, i, time_len) for i, t in enumerate(time)])
    print('Integration successully completed')
    return x


def integrate_multiple(ode_function, x0_list, time, args=()):
    """ Integrate ode for multiple initial states given by x0_list """
    x = [odeint(ode_function, x0, time, args=tuple(args)) for x0 in x0_list]
    return MultipleResultsODE(x, time, ode_function, args)

