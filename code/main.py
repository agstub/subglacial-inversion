#-------------------------------------------------------------------------------
# author: Aaron Stubblefield, Columbia University
#
# * this program inverts for (1) a basal velocity anomaly "w", (2) basal sliding coefficient
#   field "beta", or (3) sub-shelf melt rate "m", given the observed surface elevation
#   change "h_obs" (and possibly horizontal surface velocity u_obs & v_obs) by
#   solving a least-squares minimization problem
#
# * the main model assumptions are (1) Newtonian viscous ice flow, (2) a linear basal sliding law,
#   and (3) that all fields are small perturbations of a simple shear background flow.
#   These assumptions allow for efficient solution of the forward model, with 2D (map-plane)
#   Fourier transforms and convolution in time being the main operations.
#   see notes.tex for the full description of the model and numerical method
#
# * this is the main file that imports the data, calls the inverse problem solver,
#   and calls any post-processing functions (e.g., plotting)
#
# * Primary files to "playing around" with are:
#   o params.py: set the inversion options and physical/numerical parameters
#   o synthetic_data.py: create different synthetic data
#-------------------------------------------------------------------------------
from params import dim,make_movie,vel_data,delta,eps_w,eps_beta,eps_m
from synthetic_data import h_obs_synth,w_true,beta_true,m_true,h_true,u_obs_synth,v_obs_synth
from inversion import invert
from plotting import plot_movie,snapshots,discrepancy
from aux import calc_m_hydr
import numpy as np
from conj_grad import norm

# synthetic data example (see synthetic_data.py)
h_obs = h_obs_synth

if vel_data == 0:
    data = h_obs
elif vel_data == 1:
    u_obs = u_obs_synth
    v_obs = v_obs_synth
    data = np.array([h_obs,u_obs,v_obs])

# noise = norm(h_obs-h_true)
# misfit_arr = np.zeros(5)
# eps_arr = np.zeros(5)
#
# i=1
sol,fwd = invert(data) # solve the inverse problem
# misfit_arr[i] = norm(h_obs-fwd)
# eps_arr[i] = eps_w()+eps_beta()+eps_m()
#
# discrepancy(misfit_arr,eps_arr,noise)

if dim == 1:
    sol_true = w_true+beta_true+m_true
    snapshots(sol,data,sol_true)
#
elif dim == 2:
    snapshots(sol[0],data,sol[1])

if make_movie == 1:
    plot_movie(sol,fwd,data)
