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
from params import Nt,dim
from synthetic_data import h_obs_synth,w_true,beta_true
from inversion import invert
from plotting import plot_results,plot_results_joint
import os
import numpy as np

if dim ==2 :
    from synthetic_data import u_obs_synth,v_obs_synth

# synthetic data example (see synthetic_data.py)
h_obs = h_obs_synth

#u_obs = u_obs_synth

#v_obs = v_obs_synth


# print(np.min(u_obs))
# print(np.max(v_obs))
# print(np.min(v_obs))


sol,h = invert(h_obs) # solve the inverse problem

#* save a png image of the solution at each timestep
#* need to make a directory called 'pngs' first!
if os.path.isdir('pngs')==False:
    os.mkdir('pngs')    # make a directory for the results.

for i in range(Nt):
    plot_results(sol,h_obs,h_obs,i)
    #plot_results_joint(w_true,beta_true,h_obs,h_obs,u_obs,u_obs,v_obs,v_obs,i)






# Plotting Ub........
#
# def Ub(kx,ky):
#     # Horizontal velocity beta-response function
#     k = np.sqrt(kx**2 + ky**2)
#     n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
#     nx = 2*np.pi*kx
#     ny = 2*np.pi*ky
#     kap = (n/nx)**2
#
#     N = ((n-2*kap+1)*np.exp(4*n) + 2*n*(3-4*kap)*np.exp(2*n)+n+2*kap-1)
#
#     D = ((np.exp(6*n) + (4*n-1)*np.exp(4*n) - (4*n+1)*np.exp(2*n)+1))/(2*np.exp(n))
#
#     return (N/D)*(nx**2/n**3)
# #
# import matplotlib.pyplot as plt
# from matplotlib.colors import SymLogNorm
#
# kx0 = np.logspace(-9,-2,100)
# ky0 = np.logspace(-9,-2,100)
#
# kx,ky = np.meshgrid(kx0,ky0)
#
# omg = Ub(kx,ky)
#
# print(np.max(omg))
# print(np.min(omg))
#
# plt.figure(figsize=(8,6))
# plt.contourf(kx0,ky0,omg,cmap='Blues')
# plt.colorbar()
# plt.xlabel(r'$k_x$',fontsize=20)
# plt.ylabel(r'$k_y$',fontsize=20)
# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.tight_layout()
# plt.show()
# plt.close()
