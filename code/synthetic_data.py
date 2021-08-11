# this file creates synthetic data for inversion examples

from params import L,noise_level,x,y,t,Nt,Nx,Ny,inv_w,inv_beta,inv_m,vel_data,dim,ub0,t_final
from operators import forward_w,forward_m,forward_beta,forward_u,forward_v,forward_s,forward_uf,forward_vf,sg_fwd
import numpy as np

#------------------------ create synthetic data --------------------------------
sigma = 2*L/3          # standard deviation for Gaussians used in default examples
                     # (except for melt rate example, which is half of this)

# (1) VERTICAL VELOCITY ANOMALY
# *EXAMPLE 1
# Subglacial lake : Stationary Gaussian with oscillating amplitude
w_true = 50*np.exp(-0.5*(sigma**(-2))*(np.abs(x+0*L)**2+np.abs(y-0*L)**2 ))*np.cos(4*np.pi*t/(0.5*t_final))*inv_w

# *EXAMPLE 2
# Bed bump: w_b = u_b*ds/dx
#bed = np.exp(-0.5*(sigma**(-2))*(np.abs(x+2*L)**2+np.abs(y-2*L)**2 ))
# bed_x = -100*(x/(sigma**2))*np.exp(-0.5*(sigma**(-2))*(np.abs(x+0*L)**2+np.abs(y-0*L)**2 ))
# w_true = ub0*bed_x*inv_w

# (2) SLIPPERINESS ANOMALY
# Gaussian friction perturbation (constant in time)
# default is slippery spot, switch sign for sticky spot
beta_true = -8e-2*np.exp(-0.5*((sigma)**(-2))*(np.abs(x+0*L)**2+np.abs(y-0*L)**2 ))*inv_beta


# (3) MELTING ANOMALY (sub-shelf)
# travelling Gaussian melt 'wave'
xc = (20-40*t/(t_final/2))*(L/10)

m_true = 100*np.exp(-0.5*((0.5*sigma)**(-2))*(np.abs(x-xc)**2+np.abs(1*y)**2 ))*inv_m

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# solve state equation with "true solution" and add some noise

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
    u_obs_synth = u_true+noise_level*np.max(np.abs(u_true))*np.random.normal(size=(Nt,Nx,Ny))
    v_obs_synth = v_true+noise_level*np.max(np.abs(u_true))*np.random.normal(size=(Nt,Nx,Ny))

elif dim == 2:
    #elevation solution
    h_true = forward_w(w_true) + forward_beta(beta_true)
    h_obs_synth = h_true  + noise_level*np.max(np.abs(h_true))*np.random.normal(size=(Nt,Nx,Ny))

    # velocity solutions
    u_true = forward_u(w_true,beta_true)
    v_true = forward_v(w_true,beta_true)
    u_obs_synth = u_true+noise_level*np.max(np.abs(u_true))*np.random.normal(size=(Nt,Nx,Ny))
    v_obs_synth = v_true+noise_level*np.max(np.abs(u_true))*np.random.normal(size=(Nt,Nx,Ny))

from scipy.fft import ifft2,fft2
s_true = ifft2(sg_fwd(fft2(w_true))).real

# # print max values of data for sanity check...
print('Synthetic data properties:')
print('max s = '+str(np.max(np.abs(s_true))))
print('max h = '+str(np.max(np.abs(h_true))))
print('max u = '+str(np.max(np.abs(u_true))))
print('max v = '+str(np.max(np.abs(v_true))))
print('\n')
