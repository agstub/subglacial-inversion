#-------------------------------------------------------------------------------
# author: Aaron Stubblefield, Columbia University
#
# * this program inverts for (1) a basal velocity anomaly "w", (2) basal friction
#   field "beta", or (3) sub-shelf melt rate "m", given the observed surface elevation
#   change "h_obs" by solving a least-squares minimization problem
#
# * the main model assumptions are (1) Newtonian viscous ice flow, (2) a linear basal sliding law,
#   and (3) that all fields are small perturbations of a uniform background flow.
#   These assumptions allow for efficient solution of the forward model, with 2D (map-plane)
#   Fourier transforms and convolution in time being the main operations.
#   see notes.tex for the full description of the model and numerical method
#
# * this is the main file that imports the data, calls the inverse problem solver,
#   and calls any post-processing functions (e.g., plotting)
#
# * Primary files for "playing around" are:
#   o params.py: set the inversion options and physical/numerical parameters
#   o synthetic_data.py: create different synthetic data
#-------------------------------------------------------------------------------
from params import Nt
from synthetic_data import h_obs_synth
from inversion import invert
from plotting import plot_results
import os

# synthetic data example (see synthetic_data.py)
h_obs = h_obs_synth

sol,h = invert(h_obs) # solve the inverse problem

#* save a png image of the solution at each timestep
#* need to make a directory called 'pngs' first!
if os.path.isdir('pngs')==False:
    os.mkdir('pngs')    # make a directory for the results.

for i in range(Nt):
    plot_results(sol,h,h_obs,i)
