# this file contains the invert() function that defines the right-side
# vector of the normal equations and calls the conjugate-gradient solver

from params import inv_w,inv_beta,inv_m
from conj_grad import cg_solve
from operators import forward_w,forward_beta,forward_m,adjoint_w,adjoint_beta,adjoint_m
import numpy as np

def invert(h_obs):
    # invert for w, beta, or m given the observed elevation change h_obs

    sol = 0*h_obs

    print('Solving normal equations with CG....\n')
    if inv_w == 1:
        b = adjoint_w(h_obs)
        X0 = sol
        sol = cg_solve(b,X0)
        h = forward_w(sol)

    elif inv_beta == 1:
        b = adjoint_beta(h_obs)
        X0 = sol
        sol = cg_solve(b,X0)
        h = forward_beta(sol)

    elif inv_m == 1:
        b = adjoint_m(h_obs)
        X0 = sol
        sol = cg_solve(b,X0)
        h = forward_m(sol)

    return sol,h
