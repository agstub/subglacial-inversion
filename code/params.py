# this file sets the inversion options and numerical parameters/discretization

import numpy as np
from scipy.fft import fftfreq

# --------------------inversion options ---------------------------------------
# set either = 1 (on) or = 0 (off)
# inversion for one field at a time is supported
inv_w = int(1)                      # invert for w     (basal vertical velocity)
inv_beta = int(0)                   # invert for beta  (slipperiness)
inv_m = int(0)                      # invert for m     (melt rate)

# NOTE: joint inversion for w and beta benefits from supplying horizontal surface
#       velocity data---otherwise, the results tend to look very bad.

vel_data = int(0)                   # indicate whether horizontal surface velocity
                                    # data is provided as a constraint

dim = inv_w + inv_beta + inv_m      # 'dimensionality' of inverse problem


#----------------------------regularization-------------------------------------
# reguarization parameters for each inversion type
eps_w = 1e-3
eps_beta = 1e-4
eps_m = 1e-4

# Regularization options: L2 and H1 (see regularizations.py)
w_reg = 'L2'            # regularization type for w
beta_reg = 'L2'         # regularization type for beta
m_reg = 'L2'            # regularization type for m

#---------------------- physical parameters ------------------------------------

# dimensional parameters
H = 1000                    # ice thickness (m)
h_sc = 1                   # elevation anomaly scale (m)
asp = h_sc/H               # aspect ratio

slope =  0*np.pi/180       # slope of basal surface (radians)

t_sc = 1*3.154e7           # observational timescale (s)
eta = 1e13                 # Newtonian ice viscosity (Pa s)

vel_sc = h_sc/t_sc

rho_i = 917                # ice density (kg/m^3)
rho_w = 1000               # water density
g = 9.81                   # gravitational acceleration


# "background flow"
u_slope = (rho_i*g*(H**2)/(2*eta))*np.sin(slope)    # intrinsic surface velocity for inclined slope problem
u_sshear = 250/3.154e7*(1-np.abs(np.sign(slope)))   # surface velocity for simple shear problem

u_b = 200/3.154e7              # background sliding velocity (m/s)

u_h = u_b + (u_slope + u_sshear)*(1-inv_m) # background horizontal surface velocity (m/s)


# Set background drag coefficient and related parameters, depending on the bed slope
if slope > 1e-7:
    # inclined slope problem
    beta_e = 2*(eta/H)*(u_h/u_b-1) # background basal friction coeffcieint (Pa s/m)
    uzz = -rho_i*g*np.sin(slope)/eta
    uz = rho_i*g*np.sin(slope)*H/eta
else:
    # simple shear problem
    beta_e = (eta/H)*(u_h/u_b-1)
    uz = (u_h - u_b)/H
    uzz = 0

t_r = 2*eta/(rho_i*g*H*np.cos(slope))    # viscous relaxation time

# nondimensional parameters
lamda = t_sc/t_r           # process timescale relative to
                           # surface relaxation timescale

uh0 = u_h*t_sc/H           # "background" horizontal strain rate relative to
                           # vertical strain rate:
                           # U = (u/H) / (w/h0),
                           # where u = dimensional horizontal flow speed
                           #       H = ice thickness
                           #       w = vertical velocity anomaly scale
                           #       h0 = elevation anomaly scale

ub0 = u_b*t_sc/H

nu = u_b*t_sc/h_sc         # coefficient on sliding terms
                           # = sliding velocity scale/vertical velocity scale

tau = (beta_e*uz - eta*uzz)*H*t_sc/(2*eta)

beta0 = beta_e*H/(2*eta)   # friction coefficient relative to ice viscosity

delta = rho_w/rho_i-1      # density ratio

noise_level = 0.0          # noise level (scaled relative to elevation anomaly amplitude)
                           # used to create synthetic data


#---------------------- numerical parameters------------------------------------
u_wt = 0.01                      # weight on surface velocity misfit for joint inversions
h_wt = 1                         # weight on elevation misfit for joint inversion

cg_tol = 1e-9                     # stopping tolerance for conjugate gradient solver

max_cg_iter =  500                # maximum conjugate gradient iterations

# discretization parameters
Nx = 20                            # number of grid points in x-direction
Ny = 20                            # number of grid points in y-direction
Nt = 100                            # number of time steps

L = 10                      # horizontal x-y domain is an 8L x 8L square
t_final = 5                 # final time

t0 = np.linspace(0,t_final,num=Nt) # time array

dt = t0[1]

x0 = np.linspace(-4*L,4*L,num=Nx)  # x coordinate array
y0 = np.linspace(-4*L,4*L,num=Ny)  # y coordinate array
dx = np.abs(x0[1]-x0[0])           # grid size in x direction'
dy = np.abs(y0[1]-y0[0])           # grid size in y direction

kx0 =  fftfreq(Nx,dx)
ky0 =  fftfreq(Ny,dy)

# set zero frequency to small number because some of the integral kernels
# have integrable or removable singularities at the zero frequency

kx0[0] = 1e-10
ky0[0] = 1e-10


# mesh grids for physical space domain
t,x,y = np.meshgrid(t0,x0,y0,indexing='ij')

# mesh grids for frequency domain
t,kx,ky = np.meshgrid(t0,kx0,ky0,indexing='ij')

# magnitude of the wavevector
k = np.sqrt(kx**2+ky**2)
