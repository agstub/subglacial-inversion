#-------------------------------------------------------------------------------
# author: Aaron Stubblefield, Columbia University
#
# * this program inverts for (1) a basal velocity anomaly "w" or (2) basal drag
#   coefficient "beta", given the observed surface elevation change "h_obs"
#   (and possibly horizontal surface velocity u_obs & v_obs) by solving a
#   least-squares minimization problem
#
# * the main model assumptions are (1) Newtonian viscous ice flow, (2) a linear basal sliding law,
#   and (3) that all fields are small perturbations of a simple shear background flow.
#   These assumptions allow for efficient solution of the forward model, with 2D (map-plane)
#   Fourier transforms and convolution in time being the main operations.
#
# * this is the main file that imports the data, calls the inverse problem solver,
#   and calls any post-processing functions (e.g., plotting)
#
# * Primary files to "play around" with are:
#   o params.py: set the physical/numerical parameters
#   o synthetic_data.py: create different synthetic data
#-------------------------------------------------------------------------------
from synthetic_data import make_data,make_fields
from inversion import invert
from plotting import snapshots,plot_movie
import numpy as np
from conj_grad import norm
from scipy.fft import fft2,ifft2

def main(data,vel_locs,inv_w,inv_beta,eps_w,eps_beta):
    # the default examples use synthetic data (see synthetic_data.py)

    sol,fwd = invert(data,vel_locs,inv_w,inv_beta,eps_w,eps_beta) # solve the inverse problem

    mis = norm(fwd-data[0])/norm(data[0])                # normalized misfit

    return sol,fwd,mis


inv_w = 1
inv_beta = 1
eps_w = 1e1
eps_beta = 1e5

noise_level = 0.01    # noise level (scaled relative to elevation anomaly norm)

data = make_data(inv_w,inv_beta,noise_level)

vel_locs = np.ones(np.shape(data[0]),dtype=int)

sol_true = make_fields(inv_w,inv_beta)

sol,fwd,mis = main(data,vel_locs,inv_w,inv_beta,eps_w,eps_beta)

print('||h-h_obs||/||h_obs|| = '+str(mis)+' (target = '+str(noise_level)+')')

snapshots(data,fwd,sol,sol_true,inv_w,inv_beta)

plot_movie(sol,sol_true,fwd,data,inv_w,inv_beta)
