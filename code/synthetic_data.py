# this file creates synthetic data for inversion examples

from params import U,L,noise_level,x,y,t,Nt,Nx,Ny,inv_w,inv_beta,inv_m
from operators import forward_w,forward_m,forward_beta
import numpy as np

#------------------------ create synthetic data --------------------------------
sigma = L/3          # standard deviation for Gaussians used in default examples

# (1) VERTICAL VELOCITY ANOMALY
# *EXAMPLE 1
# Subglacial lake : Oscillating Gaussian
w_true = np.exp(-0.5*(sigma**(-2))*(np.abs(x+0*L)**2+np.abs(y-0*L)**2 ))*np.sin(2*np.pi*t)
w_true[np.abs(w_true)<0.1] = 0

# *EXAMPLE 2
# Bed bump: w = u*ds/dx
# bed = np.exp(-0.5*(sigma**(-2))*(np.abs(x+2*L)**2+np.abs(y-2*L)**2 ))
# bed_x = -(x/(sigma**2))*np.exp(-0.5*(sigma**(-2))*(np.abs(x+2*L)**2+np.abs(y-2*L)**2 ))
# w_true = U*bed_x

# (2) SLIPPERINESS ANOMALY
# Gaussian friction perturbation
beta_true = inv_beta*1e-7*np.exp(-0.5*(sigma**(-2))*(np.abs(x)**2+np.abs(y)**2 ))

# (3) MELTING ANOMALY (sub-shelf)
m_true = np.exp(-0.5*(sigma**(-2))*(np.abs(x)**2+np.abs(y)**2 ))
m_true[np.abs(m_true)<0.1] = 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# solve state equation with "true solution"

if inv_w == 1:
    h_true = forward_w(w_true)
elif inv_beta == 1:
    h_true = forward_beta(beta_true)
elif inv_m == 1:
    h_true = forward_m(m_true)

h_max = np.max(h_true)

# add some noise to the "true" data, as some fraction "noise_level" of the
# maximum elevation "h_max"
h_obs_synth = h_true  + (noise_level*h_max)*np.random.normal(size=(Nt,Nx,Ny))
