# this file creates synthetic data for inversion examples

from params import L,noise_level,x,y,t,Nt,Nx,Ny,inv_w,inv_beta,inv_m,t_final,dim
from operators import forward_w,forward_m,forward_beta,forward_u,forward_v
import numpy as np

#------------------------ create synthetic data --------------------------------
sigma = L/3          # standard deviation for Gaussians used in default examples
                     # (except for melt rate example, which is half of this)

# (1) VERTICAL VELOCITY ANOMALY
# *EXAMPLE 1
# Subglacial lake : Stationary Gaussian with oscillating amplitude
w_true = 15*np.exp(-0.5*(sigma**(-2))*(np.abs(x+0*L)**2+np.abs(y-0*L)**2 ))*np.sin(2*np.pi*t)

# *EXAMPLE 2
# Bed bump: w = u*ds/dx
# bed = np.exp(-0.5*(sigma**(-2))*(np.abs(x+2*L)**2+np.abs(y-2*L)**2 ))
# bed_x = -(x/(sigma**2))*np.exp(-0.5*(sigma**(-2))*(np.abs(x+2*L)**2+np.abs(y-2*L)**2 ))
# w_true = U*bed_x

# (2) SLIPPERINESS ANOMALY
# Gaussian friction perturbation (constant in time)
beta_true = 0*2e-2*np.exp(-0.5*((sigma)**(-2))*(np.abs(x-0*L)**2+np.abs(y+0*L)**2 ))

# (3) MELTING ANOMALY (sub-shelf)
# travelling Gaussian melt 'wave'
xc = 20-40*t
m_true = 300*np.exp(-0.5*((sigma/2)**(-2))*(np.abs(x-xc)**2+np.abs(y)**2 ))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# solve state equation with "true solution"

if dim==1 and inv_w == 1:
    h_true = forward_w(w_true)
elif dim==1 and inv_beta == 1:
    h_true = forward_beta(beta_true)
elif dim==1 and inv_m == 1:
    h_true = forward_m(m_true)
elif dim == 2:
    h_true = forward_w(w_true) + forward_beta(beta_true)
    u_true = forward_u(w_true,beta_true)
    v_true = forward_v(w_true,beta_true)
    u_obs_synth = u_true
    v_obs_synth = v_true


h_max = np.max(np.abs(h_true))

# add some noise to the "true" data, as some fraction "noise_level" of the
# maximum elevation "h_max"
h_obs_synth = h_true  + (noise_level*h_max)*np.random.normal(size=(Nt,Nx,Ny))
