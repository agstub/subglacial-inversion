# some aunuliary functions
from scipy.fft import ifft2,fft2
from params import rho_i,rho_w,uh0,dt,kx,ky,t_final,asp,nu,delta,H
import numpy as np
from synthetic_data import u_true,v_true

def calc_m_hydr(h):
    # calculate the melt rate, assuming purely hydrostatic ice thickness
    c = 1+1/delta
    ht = np.gradient(h,dt/t_final,axis=0)
    hx = ifft2(1j*2*np.pi*kx*fft2(h)).real
    hy = ifft2(1j*2*np.pi*ky*fft2(h)).real
    ux = ifft2(1j*2*np.pi*kx*fft2(u_true)).real
    vy = ifft2(1j*2*np.pi*ky*fft2(v_true)).real

    return -c*(ht+asp*(nu+u_true)*hx + asp*v_true*hy + asp*(1/(c*asp) + h)*(ux+vy) )
