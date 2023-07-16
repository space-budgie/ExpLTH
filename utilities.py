from sympy import diff, integrate
import numpy as np


def gauge_transform(vector_potential, position_vars, to_zero=0):
    "Transform into a gauge where the to_zeroth component of the specified vector potential is 0."
    gauge = integrate(vector_potential[to_zero],
                      (position_vars[to_zero], 0, position_vars[to_zero]))
    # to_zeroth component will be 0 by fundamental theorem of calculus
    return [vp - diff(gauge, position_vars[i]) for i, vp in enumerate(vector_potential)]


def beta_to_gamma(beta):
    return 1/np.sqrt(1-beta**2)
