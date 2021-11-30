# this file contains the invert() function that defines the right-side
# vector of the normal equations and calls the conjugate-gradient solver

from params import inv_w,inv_beta,dim,vel_data,Nx,Ny,Nt
from gps_stats import gps_locs
from conj_grad import cg_solve
from operators import (forward_w,forward_beta,adjoint_w,adjoint_beta,Hc,vel_data,
                        adjoint_Ub,adjoint_Uw,adjoint_Vb,adjoint_Vw,forward_U,forward_V,h_wt,u_wt)
import numpy as np
from scipy.fft import fft2,ifft2

def invert(data,X0):
    # invert for w, beta, or m given the observed elevation change h_obs

    print('Solving normal equations with CG....\n')
    if inv_w == 1 and dim==1:
        if vel_data == 0:
            b = adjoint_w(fft2(data))

        elif vel_data == 1:
            b = h_wt*adjoint_w(fft2(data[0]))+u_wt*(adjoint_Uw(fft2(data[1]))+adjoint_Vw(fft2(data[2])))*gps_locs

        sol = cg_solve(b,X0)
        fwd = ifft2(forward_w(sol)).real

    elif inv_beta == 1 and dim==1:
        if vel_data == 0:
            b = h_wt*adjoint_beta(fft2(data))
        elif vel_data == 1:
            b = h_wt*adjoint_beta(fft2(data[0]))+u_wt*(adjoint_Ub(fft2(data[1]))+adjoint_Vb(fft2(data[2])))*gps_locs

        sol = cg_solve(b,X0)
        fwd = ifft2(forward_beta(sol)).real

    elif dim == 2:
        # data = [h_obs,u_obs,v_obs]
        if vel_data == 1:
            b1 = h_wt*adjoint_w(fft2(data[0]))+u_wt*(adjoint_Uw(fft2(data[1]))+adjoint_Vw(fft2(data[2])))
            b2 = h_wt*adjoint_beta(fft2(data[0]))+u_wt*(adjoint_Ub(fft2(data[1]))+adjoint_Vb(fft2(data[2])))
        elif vel_data == 0:
            b1 = adjoint_w(fft2(data))
            b2 = adjoint_beta(fft2(data))

        b = np.array([b1,b2])
        sol = cg_solve(b,X0)
        h = ifft2(Hc(sol)).real
        u = ifft2(forward_U(sol[0],sol[1])).real
        v = ifft2(forward_V(sol[0],sol[1])).real
        fwd = np.array([h,u,v])

    return sol,fwd
