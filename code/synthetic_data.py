# this file creates synthetic data for inversion examples

from params import L,noise_level,x,y,t,Nt,Nx,Ny,inv_w,inv_beta,vel_data,dim,ub0,t_final,dt
from operators import forward_w,forward_beta,forward_U,forward_V,sg_fwd
import numpy as np
from conj_grad import norm
from scipy.fft import ifft2,fft2

# Temporal box function for basal drag example
def box(t):
    res=t*np.sign(0.5*t_final-t)
    res[(t>=0.25*t_final)&(t<=0.75*t_final)] = t_final/4.0
    return res

#------------------------ create synthetic data --------------------------------
sigma = 2*L/3        # standard deviation for Gaussians used in default examples
                     # (except for melt rate example, which is half of this)

# (1) VERTICAL VELOCITY ANOMALY
# *EXAMPLE 1
# Subglacial lake : Stationary Gaussian with oscillating amplitude
w_true = 50*np.exp(-0.5*(sigma**(-2))*(x**2+y**2 ))*np.sin(2*np.pi*t/t_final)*inv_w

# *EXAMPLE 2
# Bed bump: w_b = u_b*ds/dx
#bed = np.exp(-0.5*(sigma**(-2))*(np.abs(x+2*L)**2+np.abs(y-2*L)**2 ))
# bed_x = -100*(x/(sigma**2))*np.exp(-0.5*(sigma**(-2))*(np.abs(x+0*L)**2+np.abs(y-0*L)**2 ))
# w_true = ub0*bed_x*inv_w

# (2) SLIPPERINESS ANOMALY
# Gaussian friction perturbation (constant in time)
# default is slippery spot, switch sign for sticky spot
beta_true = -8e-2*np.exp(-0.5*((sigma)**(-2))*(x**2+y**2 ))*np.sin(2*np.pi*box(t)/t_final)*inv_beta

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# solve state equation with "true solution"

if dim==1 and inv_w == 1:
    #elevation solution
    h_true = forward_w(w_true)
    s_true = ifft2(sg_fwd(fft2(w_true))).real

    # velocity solutions
    u_true = forward_U(w_true,0*beta_true)
    v_true = forward_V(w_true,0*beta_true)

elif dim==1 and inv_beta == 1:
    #elevation solution
    h_true = forward_beta(beta_true)
    s_true = 0*h_true

    # velocity solutions
    u_true = forward_U(0*w_true,beta_true)
    v_true = forward_V(0*w_true,beta_true)

elif dim == 2:
    #elevation solution
    h_true = forward_w(w_true) + forward_beta(beta_true)
    h_obs_synth = h_true  + noise_level*np.max(np.abs(h_true))*np.random.normal(size=(Nt,Nx,Ny))
    s_true = ifft2(sg_fwd(fft2(w_true))).real

    # velocity solutions
    u_true = forward_U(w_true,beta_true)
    v_true = forward_V(w_true,beta_true)

# Add some noise
noise_h = np.random.normal(size=(Nt,Nx,Ny))
noise_u = np.random.normal(size=(Nt,Nx,Ny))
noise_v = np.random.normal(size=(Nt,Nx,Ny))
#
h_obs_synth = h_true  + noise_level*norm(h_true)*noise_h/norm(noise_h)
u_obs_synth = u_true  + 0.1*noise_level*norm(u_true)*noise_u/norm(noise_u)
v_obs_synth = v_true  + 0.1*noise_level*norm(v_true)*noise_v/norm(noise_v)


# print max values of data for sanity check...
# print('\n')
# print('-----------------------------------------------------------')
# print('Synthetic data properties:')
# print('max h = '+str(np.max(np.abs(h_true))))
# print('max s = '+str(np.max(np.abs(s_true))))
# print('max speed = '+str(np.max(np.sqrt(u_true**2 + v_true**2))))
# print('-----------------------------------------------------------')
# print('\n')
