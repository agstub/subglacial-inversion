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
# * Primary files to "playing around" with are:
#   o params.py: set the inversion options and physical/numerical parameters
#   o synthetic_data.py: create different synthetic data
#-------------------------------------------------------------------------------
from params import dim,make_movie,vel_data,save_sol,load_sol,noise_level,nonlin_ex,Nt
from synthetic_data import h_obs_synth,w_true,beta_true,h_true,u_obs_synth,v_obs_synth
from inversion import invert
from plotting import plot_movie,snapshots,snapshots_1D
import numpy as np
from aux import interp
from conj_grad import norm
from operators import sg_fwd
from scipy.fft import fft2,ifft2
import os

def main():
    # the default examples use synthetic data (see synthetic_data.py)
    h_obs = h_obs_synth

    if vel_data == 0:
        data = h_obs
    elif vel_data == 1:
        u_obs = u_obs_synth
        v_obs = v_obs_synth
        data = np.array([h_obs,u_obs,v_obs])

    if dim ==2 and load_sol == 1:
        wb_0 = np.load('wb_f.npy')
        beta_0 = np.load('beta_f.npy')
        X0 = np.array([wb_0,beta_0])
    elif dim ==2 and load_sol == 0:
        X0 = 0*np.array([h_obs,h_obs])
    elif dim == 1:
        X0 = 0*h_obs
    #
    sol,fwd = invert(data,X0) # solve the inverse problem

    # #
    if dim ==2 and save_sol == 1:
        print('Interpolating solutions on finer grid...')
        wb_f = interp(sol[0])
        beta_f = interp(sol[1])
        print('Saving solutions...')
        np.save('wb_f.npy',wb_f)
        np.save('beta_f.npy',beta_f)

    if dim == 1:
        sol_true = w_true+beta_true
        if nonlin_ex != 1:
            snapshots(data,sol,sol_true)
        else:
            finn = 'dog'
            #snapshots_1D(data,fwd,sol,sol_true)


    elif dim == 2:
        snapshots(data,sol[0],sol[1])

    if make_movie == 1:
        plot_movie(sol,fwd,data)


    dsc = norm(fwd-h_obs)/norm(h_obs)

    print('|h-h_obs|/|h_obs| = '+str(dsc)+' (target = '+str(noise_level)+')')

    # save solution as .npy file
    if save_sol == 1:
        np.save('sol.npy',sol)

    return sol,fwd

sol,fwd = main()

if os.path.isdir('pngs_nl')==False:
    os.mkdir('pngs_nl')    # make a directory for the results.

for i in range(Nt):
    snapshots_1D(h_obs_synth,fwd,sol,w_true,i)
