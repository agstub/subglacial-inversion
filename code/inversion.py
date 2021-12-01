# this file contains the invert() function that defines the right-side
# vector of the normal equations and calls the conjugate-gradient solver

from conj_grad import cg_solve
from operators import (forward_w,forward_beta,adjoint_w,adjoint_beta,Hc,
                        adjoint_Ub,adjoint_Uw,adjoint_Vb,adjoint_Vw,forward_U,forward_V,h_wt,u_wt)
import numpy as np
from scipy.fft import fft2,ifft2

def invert(data,vel_locs,inv_w,inv_beta,eps_w,eps_beta):
    # invert for w, beta, or m given the observed elevation change h_obs
    # and possibly horizontal surface velocity (u_obs,v_obs) defined at locations vel_locs
    #
    # data = [h_obs,u_obs,v_obs]
    

    dim = inv_w + inv_beta
    vel_data = np.max(vel_locs)

    print('Solving normal equations with CG....\n')
    if inv_w == 1 and dim==1:
        if vel_data == 0:
            b = adjoint_w(fft2(data[0]))

        elif vel_data == 1:
            b = h_wt*adjoint_w(fft2(data[0]))+u_wt*(adjoint_Uw(fft2(data[1]))+adjoint_Vw(fft2(data[2])))*vel_locs

        sol = cg_solve(b,inv_w,inv_beta,eps_w,eps_beta,vel_locs)
        fwd = ifft2(forward_w(sol)).real

    elif inv_beta == 1 and dim==1:
        if vel_data == 0:
            b = h_wt*adjoint_beta(fft2(data[0]))
        elif vel_data == 1:
            b = h_wt*adjoint_beta(fft2(data[0]))+u_wt*(adjoint_Ub(fft2(data[1]))+adjoint_Vb(fft2(data[2])))*vel_locs

        sol = cg_solve(b,inv_w,inv_beta,eps_w,eps_beta,vel_locs)
        fwd = ifft2(forward_beta(sol)).real

    elif dim == 2:
        if vel_data == 1:
            b1 = h_wt*adjoint_w(fft2(data[0]))+u_wt*(adjoint_Uw(fft2(data[1]))+adjoint_Vw(fft2(data[2])))
            b2 = h_wt*adjoint_beta(fft2(data[0]))+u_wt*(adjoint_Ub(fft2(data[1]))+adjoint_Vb(fft2(data[2])))
        elif vel_data == 0:
            b1 = adjoint_w(fft2(data[0]))
            b2 = adjoint_beta(fft2(data[0]))

        b = np.array([b1,b2])
        sol = cg_solve(b,inv_w,inv_beta,eps_w,eps_beta,vel_locs)
        h = ifft2(Hc(sol)).real
        u = ifft2(forward_U(sol[0],sol[1])).real
        v = ifft2(forward_V(sol[0],sol[1])).real
        fwd = np.array([h,u,v])

    return sol,fwd
