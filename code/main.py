#-------------------------------------------------------------------------------
# author: Aaron Stubblefield, Columbia University
#
# * this program inverts for (1) a basal velocity anomaly "w", (2) basal friction
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
from params import Nt,vel_data,dim,delta
from synthetic_data import h_obs_synth,w_true,beta_true,m_true,h_true,u_true,v_true
from inversion import invert
from plotting import plot_results,plot_results_joint,snapshots,plot_RB
from aux import calc_m_hydr
import os
import numpy as np

# synthetic data example (see synthetic_data.py)
h_obs = h_obs_synth

if vel_data == 0:
    data = h_obs

if vel_data == 1:
    from synthetic_data import u_obs_synth,v_obs_synth
    u_obs = u_obs_synth
    v_obs = v_obs_synth
    data = np.array([h_obs,u_obs,v_obs])


sol,fwd = invert(data) # solve the inverse problem

#* save a png image of the solution at each timestep
#* need to make a directory called 'pngs' first!
if os.path.isdir('pngs')==False:
    os.mkdir('pngs')    # make a directory for the results.

if dim == 2:
    w = sol[0]
    beta = sol[1]
    h,u,v = fwd

for i in range(Nt):
    if dim == 1 :
        plot_results(sol,h_obs,h_obs,i)
    elif dim == 2:
        plot_results_joint(w,beta,h,h_obs,u,u_obs,v,v_obs,i)
