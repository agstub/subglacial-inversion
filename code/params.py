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

u_wt = lambda : 0.01                # weight on surface velocity misfit for joint inversions
h_wt = lambda : 1                   # weight on elevation misfit for joint inversion

dim = inv_w + inv_beta + inv_m      # 'dimensionality' of inverse problem

make_movie = int(0)                 # make movie of simulation (png at each timestep)

#----------------------------regularization-------------------------------------
# reguarization parameters for each inversion type
eps_w = lambda: 1e-4*inv_w  # range: 1e-1 to 1e-5
eps_beta = lambda: 1e-3*inv_beta
eps_m = lambda: 1e-5*inv_m

# Regularization options: L2 and H1 (see regularizations.py)
w_reg = 'L2'            # regularization type for w
beta_reg = 'L2'         # regularization type for beta
m_reg = 'L2'            # regularization type for m

#---------------------- physical parameters ------------------------------------

# dimensional parameters
H = 1000                    # ice thickness (m)
h_sc = 1                   # elevation anomaly scale (m)
asp = h_sc/H               # aspect ratio

L = 10                      # horizontal x-y domain is an 8L x 8L square (horizontal length scale = H)
t_final = 10                # final time

t_sc = 1*3.154e7           # observational timescale (s)
eta = 1e13                 # Newtonian ice viscosity (Pa s)

vel_sc = h_sc/t_sc

rho_i = 917                # ice density (kg/m^3)
rho_w = 1000               # water density
g = 9.81                   # gravitational acceleration


# "background flow" (default examples)
if inv_m == 0:
    H = 1000                    # ice thickness (m)

    slope =  0.2*(np.pi/180)       # slope of basal surface (radians): default 0.2

    uh_slope = (rho_i*g*np.sin(slope)*(H**2)/(2*eta))*np.abs(np.sign(slope))   # intrinsic surface velocity for inclined slope problem

    uh_sshear = 250/3.154e7*(1-np.abs(np.sign(slope)))   # surface velocity for simple shear problem

    ub = 200/3.154e7                             # background sliding velocity (m/s)

    uh = ub*np.abs(np.sign(slope)) + uh_sshear + uh_slope           # background horizontal surface velocity (m/s)

elif inv_m == 1:
    H = 500                    # ice thickness (m)
    slope = 0
    ub = 2000/3.154e7
    uh = 2000/3.154e7

# Set background drag coefficient and related parameters, depending on the bed slope
if slope > 1e-7:
    # inclined slope problem
    beta_e = 2*(eta/H)*(uh/ub-1) # background basal friction coeffcieint (Pa s/m)
    uzz = -rho_i*g*np.sin(slope)/eta
    uz = rho_i*g*np.sin(slope)*H/eta
else:
    # simple shear problem
    beta_e = (eta/H)*(uh/ub-1)
    uz = (uh - ub)/H
    uzz = 0

## sanity check printing ....
print('Background state properties:')
print('bed slope = '+str(slope*180/np.pi)+' deg.')
print('u_h = '+str(uh*3.154e7))
print('u_b = '+str(ub*3.154e7))
print('beta = '+"{:.1E}".format(beta_e))
print('\n')


t_r = 2*eta/(rho_i*g*H*np.cos(slope))    # viscous relaxation time

# nondimensional parameters
lamda = t_sc/t_r           # process timescale relative to
                           # surface relaxation timescale

uh0 = uh*t_sc/H           # "background" horizontal strain rate relative to
                           # vertical strain rate:
                           # U = (u/H) / (w/h0),
                           # where u = dimensional horizontal flow speed
                           #       H = ice thickness
                           #       w = vertical velocity anomaly scale
                           #       h0 = elevation anomaly scale

ub0 = ub*t_sc/H

nu = ub*t_sc/h_sc         # coefficient on sliding terms
                           # = sliding velocity scale/vertical velocity scale

tau = (beta_e*uz - eta*uzz)*H*t_sc/(2*eta)

beta0 = beta_e*H/(2*eta)   # friction coefficient relative to ice viscosity

delta = rho_w/rho_i-1      # density ratio

noise_level = 0.01         # noise level (scaled relative to elevation anomaly amplitude)
                           # used to create synthetic data


#---------------------- numerical parameters------------------------------------
cg_tol = 1e-10                    # stopping tolerance for conjugate gradient solver

max_cg_iter =  500               # maximum conjugate gradient iterations

# discretization parameters
Nx = 20                            # number of grid points in x-direction
Ny = 20                            # number of grid points in y-direction
Nt = 200                            # number of time steps

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
