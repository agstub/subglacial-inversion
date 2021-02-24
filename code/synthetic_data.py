# this file creates synthetic data for inversion examples

from params import U,L,noise_level,x,y,t,Nt,Nx,Ny,inv_w,inv_beta
from operators import forward
import numpy as np

#------------------------ create synthetic data --------------------------------

# create synthetic velocity anomaly "true solution":
# a Gaussian bump with time-oscillating amplitude
sigma = 2*L/6.0

# Oscillationg Gaussian
w_lake = inv_w*np.exp(-0.5*(sigma**(-2))*(np.abs(x+0*L)**2+np.abs(y-0*L)**2 ))*np.sin(t)
w_lake[np.abs(w_lake)<0.1] = 0


# Gaussian bed bump: w = u*ds/dx
bed = 0*np.exp(-0.5*(sigma**(-2))*(np.abs(x+2*L)**2+np.abs(y-2*L)**2 ))
bed_x = -0*(x/(sigma**2))*np.exp(-0.5*(sigma**(-2))*(np.abs(x+2*L)**2+np.abs(y-2*L)**2 ))
w_bed = U*bed_x

w_true = w_lake + w_bed

# Gaussian friction perturbation
beta_true = inv_beta*np.exp(-0.5*(sigma**(-2))*(np.abs(x-0*L)**2+np.abs(y+0*L)**2 ))
beta_true[beta_true<0.1] = 0

#solve state equation with "true solution"
h_true = forward(w_true,beta_true)

h_max = np.max(h_true)

# add some noise to the "true" data, as some fraction "noise_level" of the
# maximum elevation "h_max"
h_obs_synth = h_true  + (noise_level*h_max)*np.random.normal(size=(Nt,Nx,Ny))
