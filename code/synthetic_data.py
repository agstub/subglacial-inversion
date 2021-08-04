# this file creates synthetic data for inversion examples

from params import L,noise_level,x,y,t,Nt,Nx,Ny,inv_w,inv_beta,inv_m,vel_data,dim,theta,t_final
from operators import forward_w,forward_m,forward_beta,forward_u,forward_v,forward_s,forward_uf,forward_vf
import numpy as np

#------------------------ create synthetic data --------------------------------
sigma = L/3          # standard deviation for Gaussians used in default examples
                     # (except for melt rate example, which is half of this)

# (1) VERTICAL VELOCITY ANOMALY
# *EXAMPLE 1
# Subglacial lake : Stationary Gaussian with oscillating amplitude
w_true = 15*np.exp(-0.5*(sigma**(-2))*(np.abs(x+0*L)**2+np.abs(y-0*L)**2 ))*np.sin(2*np.pi*t)*inv_w

# *EXAMPLE 2
# Bed bump: w = u*ds/dx
# bed = np.exp(-0.5*(sigma**(-2))*(np.abs(x+2*L)**2+np.abs(y-2*L)**2 ))
# bed_x = (x/(sigma**2))*np.exp(-0.5*(sigma**(-2))*(np.abs(x+0*L)**2+np.abs(y-0*L)**2 ))
# w_true = theta*bed_x

# (2) SLIPPERINESS ANOMALY
# Gaussian friction perturbation (constant in time)
# default is slippery spot, switch sign for sticky spot
beta_true = -8e-3*np.exp(-0.5*((sigma)**(-2))*(np.abs(x+0*L)**2+np.abs(y-0*L)**2 ))*inv_beta


# (3) MELTING ANOMALY (sub-shelf)
# travelling Gaussian melt 'wave'
xc = 0#20-60*t/t_final
m_true = 10*np.exp(-0.5*((sigma)**(-2))*(np.abs(1*x-xc)**2+np.abs(1*y)**2 ))*inv_m

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# solve state equation with "true solution"

if dim==1 and inv_w == 1:
    #elevation solution
    h_true = forward_w(w_true)
    h_obs_synth = h_true  + noise_level*np.max(np.abs(h_true))*np.random.normal(size=(Nt,Nx,Ny))

    # velocity solutions
    u_true = forward_u(w_true,0*beta_true)
    v_true = forward_v(w_true,0*beta_true)
    u_obs_synth = u_true+noise_level*np.max(np.abs(u_true))*np.random.normal(size=(Nt,Nx,Ny))
    v_obs_synth = v_true+noise_level*np.max(np.abs(u_true))*np.random.normal(size=(Nt,Nx,Ny))

elif dim==1 and inv_beta == 1:
    #elevation solution
    h_true = forward_beta(beta_true)
    h_obs_synth = h_true  + noise_level*np.max(np.abs(h_true))*np.random.normal(size=(Nt,Nx,Ny))

    # velocity solutions
    u_true = forward_u(0*w_true,beta_true)
    v_true = forward_v(0*w_true,beta_true)
    u_obs_synth = u_true+noise_level*np.max(np.abs(u_true))*np.random.normal(size=(Nt,Nx,Ny))
    v_obs_synth = v_true+noise_level*np.max(np.abs(u_true))*np.random.normal(size=(Nt,Nx,Ny))
elif inv_m == 1:
    #elevation solution
    h_true = forward_m(m_true)
    s_true = forward_s(m_true)
    h_obs_synth = h_true  + noise_level*np.max(np.abs(h_true))*np.random.normal(size=(Nt,Nx,Ny))

    # velocity solutions
    u_true = forward_uf(h_true,s_true)
    v_true = forward_vf(h_true,s_true)

elif dim == 2:
    #elevation solution
    h_true = forward_w(w_true) + forward_beta(beta_true)
    h_obs_synth = h_true  + noise_level*np.max(np.abs(h_true))*np.random.normal(size=(Nt,Nx,Ny))

    # velocity solutions
    u_true = forward_u(w_true,beta_true)
    v_true = forward_v(w_true,beta_true)
    u_obs_synth = u_true+noise_level*np.max(np.abs(u_true))*np.random.normal(size=(Nt,Nx,Ny))
    v_obs_synth = v_true+noise_level*np.max(np.abs(u_true))*np.random.normal(size=(Nt,Nx,Ny))
