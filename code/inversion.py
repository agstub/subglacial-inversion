# this file contains the invert() function that defines the right-side
# vector of the normal equations and calls the conjugate-gradient solver

from params import inv_w,inv_beta,inv_m,dim,vel_data
from conj_grad import cg_solve
from operators import (forward_w,forward_beta,forward_m,adjoint_w,adjoint_beta,adjoint_m,Hc,
                        adjoint_Ub,adjoint_Uw,adjoint_Vb,adjoint_Vw,forward_u,forward_v,h_wt,u_wt)
import numpy as np

def invert(data):
    # invert for w, beta, or m given the observed elevation change h_obs

    print('Solving normal equations with CG....\n')
    if inv_w == 1 and dim==1:
        b = adjoint_w(data)
        X0 = 0*b
        sol = cg_solve(b,X0)
        fwd = forward_w(sol)

    elif inv_beta == 1 and dim==1:
        b = adjoint_beta(data)
        X0 = 0*b
        sol = cg_solve(b,X0)
        fwd = forward_beta(sol)

    elif inv_m == 1:
        b = adjoint_m(data)
        X0 = 0*b
        sol = cg_solve(b,X0)
        fwd = forward_m(sol)
        # need to add velocity data option here

    elif dim == 2:
        # data = [h_obs,u_obs,v_obs]
        if vel_data == 1:
            b1 = h_wt*adjoint_w(data[0])+u_wt*(adjoint_Uw(data[1])+adjoint_Vw(data[2]))
            b2 = h_wt*adjoint_beta(data[0])+u_wt*(adjoint_Ub(data[1])+adjoint_Vb(data[2]))
        elif vel_data == 0:
            b1 = adjoint_w(data)
            b2 = adjoint_beta(data)

        b = np.array([b1,b2])
        X0 = 0*b
        sol = cg_solve(b,X0)
        h = Hc(sol)
        u = forward_u(sol[0],sol[1])
        v = forward_v(sol[0],sol[1])
        fwd = np.array([h,u,v])

    return sol,fwd
