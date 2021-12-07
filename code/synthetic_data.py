# this file creates synthetic data for inversion examples

from params import L,x,y,t,Nt,Nx,Ny,ub0,t_final,dt
from aux import nonlin_ex
from operators import forward_w,forward_beta,forward_U,forward_V,sg_fwd
import numpy as np
from conj_grad import norm
from scipy.fft import ifft2,fft2

# Temporal box function for basal drag example
def box(t):
    res=t*np.sign(0.5*t_final-t)
    res[(t>=0.25*t_final)&(t<=0.75*t_final)] = t_final/4.0
    return res


def make_fields(inv_w,inv_beta):
    dim = inv_w + inv_beta

    sigma = 2*L/3        # standard deviation for Gaussians used in default examples
                         # (except for melt rate example, which is half of this)

    # (1) VERTICAL VELOCITY ANOMALY
    # *EXAMPLE 1
    # Subglacial lake : Stationary Gaussian with oscillating amplitude
    if nonlin_ex != 1:
        w_true = 5*np.exp(-0.5*(sigma**(-2))*(x**2+y**2 ))*np.sin(2*np.pi*t/t_final)*inv_w
    else:
    # *EXAMPLE 2
    # velocity from nonlinear subglacial lake model output
    # (sawtooth volume change timeseries)
        w_true = np.load('../input/wb_true.npy')*inv_w

    # (2) SLIPPERINESS ANOMALY
    # Gaussian friction perturbation (constant in time)
    # default is slippery spot, switch sign for sticky spot
    beta_true = -8e-2*np.exp(-0.5*((sigma)**(-2))*(x**2+y**2 ))*np.sin(2*np.pi*box(t)/t_final)*inv_beta

    if dim == 1 and inv_w == 1:
        fields = w_true
    elif dim == 1 and inv_beta == 1:
        fields = beta_true
    elif dim == 2:
        fields = [w_true,beta_true]

    return fields


def make_data(inv_w,inv_beta,noise_level):
    dim = inv_w + inv_beta

    # solve state equation with "true solution"

    if dim==1 and inv_w == 1:
        w_true = make_fields(inv_w,inv_beta)

        #elevation solution
        if nonlin_ex != 1:
            h_true = ifft2(forward_w(w_true)).real
            s_true = ifft2(sg_fwd((w_true))).real

            # velocity solutions
            u_true = ifft2(forward_U(w_true,0*w_true)).real
            v_true = ifft2(forward_V(w_true,0*w_true)).real
        else:
            h_true = np.load('../input/h_true.npy')
            s_true = 0*h_true
            u_true = 0*x
            v_true = 0*x

    elif dim==1 and inv_beta == 1:
        beta_true = make_fields(inv_w,inv_beta)

        #elevation solution
        h_true = ifft2(forward_beta(beta_true)).real
        s_true = 0*h_true

        # velocity solutions
        u_true = ifft2(forward_U(0*beta_true,beta_true)).real
        v_true = ifft2(forward_V(0*beta_true,beta_true)).real

    elif dim == 2:
        w_true,beta_true = make_fields(inv_w,inv_beta)
        #elevation solution
        h_true = ifft2(forward_w(w_true) + forward_beta(beta_true)).real
        s_true = ifft2(sg_fwd(w_true)).real

        # velocity solutions
        u_true = ifft2(forward_U(w_true,beta_true)).real
        v_true = ifft2(forward_V(w_true,beta_true)).real

    # Add some noise
    noise_h = np.random.normal(size=(Nt,Nx,Ny))
    noise_u = np.random.normal(size=(Nt,Nx,Ny))
    noise_v = np.random.normal(size=(Nt,Nx,Ny))

    h_obs = h_true  + noise_level*norm(h_true)*noise_h/norm(noise_h)
    u_obs = u_true  + 0.1*noise_level*norm(u_true)*noise_u/norm(noise_u)
    v_obs = v_true  + 0.1*noise_level*norm(v_true)*noise_v/norm(noise_v)


    return h_obs,u_obs,v_obs

# print max values of data for sanity check...
# print('\n')
# print('-----------------------------------------------------------')
# print('Synthetic data properties:')
# print('max h = '+str(np.max(np.abs(h_true))))
# print('max s = '+str(np.max(np.abs(s_true))))
# print('max speed = '+str(np.max(np.sqrt(u_true**2 + v_true**2))))
# print('-----------------------------------------------------------')
# print('\n')

# # sanity check plotting
# import matplotlib.pyplot as plt
# levels = np.linspace(-0.5,0.5,9)
# plt.contourf(h_obs_synth[100,:,:].T,cmap='coolwarm',vmin=levels[0],vmax=levels[-1],extend='both',levels=levels)
# plt.colorbar()
# plt.show()
# plt.close()
