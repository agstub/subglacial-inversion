#-------------------------------------------------------------------------------
# author: Aaron Stubblefield, Columbia University
#
# * this program inverts for (1) a basal velocity anomaly "w" or (2) basal drag
#   coefficient "beta", given the observed surface elevation change "h_obs"
#   (and possibly horizontal surface velocity u_obs & v_obs) by solving a
#   least-squares minimization problem
#
# * the main model assumptions are (1) Newtonian viscous ice flow, (2) a linear basal sliding law,
#   and (3) that all fields are small perturbations of a simple background flow.
#   These assumptions allow for efficient solution of the forward model, with 2D (map-plane)
#   Fourier transforms and convolution in time being the main operations.
#
# * this is the main file that can be used to import the data, call the inverse problem solver,
#   and do any post-processing (e.g., plotting)
#
# * Primary files to "play around" with are:
#   o params.py: set the physical/numerical parameters
#   o synthetic_data.py: create different synthetic data
#-------------------------------------------------------------------------------
from inversion import invert
from conj_grad import norm

def main(data,vel_locs,inv_w,inv_beta,eps_w,eps_beta):
    # the default examples use synthetic data (see synthetic_data.py)

    # solve the inverse problem
    sol,fwd = invert(data,vel_locs,inv_w,inv_beta,eps_w,eps_beta)

    # compute normalized elevation misfit
    mis = norm(fwd-data[0])/norm(data[0])

    return sol,fwd,mis


# see notebooks for examples of how to import synthetic data, post-process, etc.
