# this file sets the inversion options and numerical parameters/discretization

import numpy as np
from scipy.fft import fftfreq

# --------------------inversion options ---------------------------------------
# set either = 1 (on) or = 0 (off)
inv_w = int(1)                      # invert for w
inv_beta = int(0)                   # invert for beta

inv_couple = inv_w*inv_beta         # 1 for simultaneous inversion, zero otherwise

#----------------------------regularization-------------------------------------
eps_w = 1e-2 * inv_w                 # w L2 regularization parameter
eps_beta = 1e-2 * inv_beta           # grad(beta) L2 regularization parameter

# Regularization options: L2 and H1 (see regularizations.py)
w_reg = 'L2'            # regularization type for w
beta_reg = 'L2'         # regularization type for beta


#---------------------- physical parameters ------------------------------------

lamda = 1                  # filling/draining timescale relative to
                           # surface relaxation timescale
                           # see notes for definition

U = 1                      # "background" horizontal strain rate relative to
                           # vertical strain rate:
                           # U = (u/H) / (w/h0),
                           # where u = dimensional horizontal flow speed
                           #       H = ice thickness
                           #       w = vertical velocity anomaly scale
                           #       h0 = elevation anomaly scale
                           # see notes for all definitions

beta0 = 0                  # ~bed frictional coefficient relative to linearized
                           # ice viscosity. see notes for definition

noise_level = 0.01         # noise level (scaled relative to elevation anomaly amplitude)
                           # used to create synthetic data

#---------------------- numerical parameters------------------------------------
cg_tol = 1e-5                      # stopping tolerance for conjugate gradient solver

max_cg_iter =  1000                # maximum conjugate gradient iterations

# discretization parameters
Nx = 20                            # number of grid points in x-direction
Ny = 20                            # number of grid points in y-direction
Nt = 100                           # number of time steps

L = 10                             # horizontal x-y domain is an 8L x 8L square
t_final = 2*np.pi                  # final time

t0 = np.linspace(0,t_final,num=Nt) # time array

dt = t0[1]

x0 = np.linspace(-4*L,4*L,num=Nx)  # x coordinate array
y0 = np.linspace(-4*L,4*L,num=Ny)  # y coordinate array
dx = np.abs(x0[1]-x0[0])           # grid size in x direction'
dy = np.abs(y0[1]-y0[0])           # grid size in y direction

kx0 =  fftfreq(Nx,dx)
ky0 =  fftfreq(Ny,dy)

# set zero frequency to small number because the integral kernels
# have an integrable singularity at the zero frequency

kx0[0] = 1e-10
ky0[0] = 1e-10

# mesh grids for physical space domain
t,x,y = np.meshgrid(t0,x0,y0,indexing='ij')

# mesh grids for frequency domain
t,kx,ky = np.meshgrid(t0,kx0,ky0,indexing='ij')

# magnitude of the wavevector
k = np.sqrt(kx**2+ky**2)
