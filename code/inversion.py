# this file contains the invert() function that defines the right-side
# vector of the normal equations and calls the conjugate-gradient solver

from params import inv_w,inv_beta,inv_couple
from conj_grad import cg_solve
from operators import forward,adjoint_w,adjoint_beta
from synthetic_data import w_bed
import numpy as np

def invert(h_obs,w_1,beta_1):
    # invert for w and/or beta given the observed elevation change h_obs
    # and initial guess w_1 and beta_1.

    w = w_1                      # initiliaze the inverse solution
    beta = beta_1                # initiliaze the inverse solution

    # define the right-side vector in the normal equations
    # the velocity anomaly contribution from the bed is subtracted from the
    # left-side of the equation if the bed elevation is provided in synthetic_data.py
    print('Solving normal equations with CG....\n')
    if inv_couple == 1:
        bw =  adjoint_w(h_obs) - adjoint_w(forward(w_bed,0*beta))
        bb = adjoint_beta(h_obs)- adjoint_beta(forward(w_bed,0*beta))
        b = np.array([bw,bb])
        X0 = np.array([w_1,beta_1])
        w,beta = cg_solve(b,X0)
    elif inv_w == 1:
        b = adjoint_w(h_obs) - adjoint_w(forward(0*w,beta))- adjoint_w(forward(w_bed,0*beta))
        X0 = w_1
        w = cg_solve(b,X0)
    elif inv_beta == 1:
        b = adjoint_beta(h_obs)- adjoint_beta(forward(w,0*beta))
        X0 = beta_1
        beta = cg_solve(b,X0)

    h = forward(w+w_bed,beta)


    return w,beta,h
