#-------------------------------------------------------------------------------
# author: Aaron Stubblefield, Columbia University
#
# * this program inverts for a basal velocity anomaly "w" and/or basal friction
#   field "beta" given the observed surface elevation change "h_obs" by solving
#   a least-squares minimization problem
#
# * the main model assumptions are (1) linear viscous ice flow, (2) a linear basal sliding law,
#   and (3) that all fields are small perturbations of a uniform background flow.
#   These assumptions allow for efficient solution of the forward model, with 2D (map-plane)
#   Fourier transforms and convolution in time being the main operations.
#   see notes.tex for the full description of the model and numerical method
#
# * this is the main file that imports the data, calls the inverse problem solver,
#   and calls any post-processing functions (e.g., plotting)
#
# * simultaneous inversion for w and beta provides poor solutions (since the problem is very
#   underdetermined) **unless** more data is available (not implemented)
#
# * Future implementation ideas:
#   o (0) Restriction of data to a subset of points (x,y,t)
#   o (1) incorporate surface velocity data to better constrain simultaneous inversion
#         (requires deriving more forward model operators)
#   o (2) add more regularization options like total variation or L1
#         (requires Newton method, not included here yet for lack of use)
#
# * Primary files for "playing around" are:
#   o params.py: set the inversion options and numerical parameters
#   o synthetic_data.py: create different synthetic data
#-------------------------------------------------------------------------------

from params import inv_w,inv_beta,Nt
from synthetic_data import h_obs_synth,w_true,beta_true
from inversion import invert
from plotting import plot_results

# synthetic data example (see synthetic_data.py)
h_obs = h_obs_synth

# initial guess for inversion:
# * default is zero, unless inverting for only one of the parameters.
# * if w or beta is not being inverted for, the "true" value (can just be zero)
#   needs to be passed to the foward model solver

w_1 = (1-inv_w)*w_true
beta_1 = (1-inv_beta)*beta_true

w_inv,beta_inv,h = invert(h_obs,w_1,beta_1) # solve the inverse problem

#* save a png image of the solution at each timestep
#* need to make a directory called 'pngs' first!
for i in range(Nt):
    plot_results(h,h_obs,w_inv,beta_inv,i)
