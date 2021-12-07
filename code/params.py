# this file sets the physical and numerical parameters

import numpy as np
from scipy.fft import fftfreq
from aux import nonlin_ex

# --------------------joint inversion options ----------------------------------
u_wt =  1e-1                # weight on surface velocity misfit for joint inversions
h_wt =  1                   # weight on elevation misfit for joint inversion

#----------------------------regularization-------------------------------------
# set default reguarization types

# Regularization options: L2 and H1 (see regularizations.py)
w_reg = 'H1'            # regularization type for w
beta_reg = 'H1'         # regularization type for beta

#---------------------- physical parameters ------------------------------------

# dimensional parameters
h_sc = 1                    # elevation anomaly scale (m)
H = 1000                    # ice thickness (m)
asp = h_sc/H                # aspect ratio
L = 10                      # horizontal x-y domain is an 8L x 8L square (horizontal length scale = H)
t_final = 10                # final time
t_sc = 1*3.154e7            # observational timescale (s)
eta = 1e13                  # Newtonian ice viscosity (Pa s)
vel_sc = h_sc/t_sc          # vertical velocity scale
rho_i = 917                 # ice density (kg/m^3)
g = 9.81                    # gravitational acceleration

#-----------------"background flow" (default examples)--------------------------

# slope of basal surface (radians): default 0.2 deg
slope =  0.2*(np.pi/180.0)*(1-nonlin_ex)

# background sliding velocity (m/s)
ub = (200/3.154e7)*(1-nonlin_ex)

# background horizontal surface velocity (m/s)
uh = (ub + rho_i*g*np.sin(slope)*(H**2)/(2*eta))*(1-nonlin_ex)


# Set background drag coefficient and related parameters, depending on the bed slope
if nonlin_ex != 1:
    beta_e = 2*(eta/H)*(uh/ub-1)     # background basal friction coeffcieint (Pa s/m)
else:
    beta_e = 5.0e9

uzz = -rho_i*g*np.sin(slope)/eta # second derivative of basal horizontal velocity
uz = rho_i*g*np.sin(slope)*H/eta # first derivative of basal horizontal velocity

tau_d = -(beta_e*uz - eta*uzz)   # dimensional vertical stress gradient at base

#----------------------Derived parameters---------------------------------------
t_r = 2*eta/(rho_i*g*H*np.cos(slope))                  # viscous relaxation time

# nondimensional parameters
lamda = t_sc/t_r           # process timescale relative to
                           # surface relaxation timescale

uh0 = uh*t_sc/H            # "background" horizontal strain rate at surface relative to
                           # ~vertical strain rate at surface:
                           # U = (uh/H) / (w/h0),
                           # where uh = horizontal flow speed at surface
                           #       H = ice thickness
                           #       w = vertical velocity anomaly scale
                           #       h0 = elevation anomaly scale

ub0 = ub*t_sc/H            # same as above^ but for the base

nu = ub*t_sc/h_sc          # coefficient on sliding terms
                           # = sliding velocity scale/vertical velocity scale

tau = tau_d*H*t_sc/(2*eta) # basal stress gradient parameter


beta0 = beta_e*H/(2*eta)   # friction coefficient relative to ice viscosity

#---------------------- numerical parameters------------------------------------
cg_tol = 1e-4               # stopping tolerance for conjugate gradient solver

max_cg_iter =  1000         # maximum conjugate gradient iterations

# discretization parameters
Nx = 101                    # number of grid points in x-direction
Ny = 101                    # number of grid points in y-direction
Nt = 100*(1-nonlin_ex)+200*nonlin_ex # number of time steps

# Note: we use a higher resolution for the nonlinear example due to the shorter
# oscillation period

t0 = np.linspace(0,t_final,num=Nt) # time array

dt = t0[1]

x0 = np.linspace(-4*L,4*L,num=Nx)  # x coordinate array
y0 = np.linspace(-4*L,4*L,num=Ny)  # y coordinate array
dx = np.abs(x0[1]-x0[0])           # grid size in x direction'
dy = np.abs(y0[1]-y0[0])           # grid size in y direction

# frequency
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


# #-------------------------------------------------------------------------------
#sanity check printing ....
# print('----------------Background state properties----------------')
# print('Dimensional parameters:')
# print('bed slope = '+str(slope*180/np.pi)+' deg.')
# print('tau_dim = '+str(-(beta_e*uz - eta*uzz))+' Pa/m')
# print('t_r = '+str(t_r/3.154e7)+' yr')
# print('u_h = '+str(uh*3.154e7)+' m/yr')
# print('u_b = '+str(ub*3.154e7)+' m/yr')
# print('beta = '+"{:.1E}".format(beta_e)+' Pa s/m')
# print('\n')
# print('Nondimensional parameters:')
# print('lambda = '+str(lamda))
# print('tau = '+"{:.1E}".format(tau))
# print('beta0 = '+"{:.1E}".format(beta0))
# print('ub0 = '+str(ub0))
# print('uh0 = '+str(uh0))
# print('nu = '+str(nu))
