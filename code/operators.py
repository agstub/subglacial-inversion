# this file defines the
# * (1) forward operators,
# * (2) adjoint operators, and
# * (3) the operator appearing on the left-side of the normal equations
#       (composition of forward and adjoint operators plus regularization terms)

import numpy as np
from kernel_fcns import R,T,F,Rf,B,Uw,Uh,Ub,Vw,Vh,Vb
from params import (theta,lamda,eps_beta,eps_w,eps_m,w_reg,beta_reg,m_reg,t,k,kx,ky,
                    inv_w,inv_beta,inv_m,delta,Nt,xi)
from scipy.signal import fftconvolve
from scipy.fft import ifft2,fft2
from regularizations import reg

def forward_w(w):
    # forward operator for basal vertical velocity w
    # returns the data (elevation) h

    w_ft = fft2(w)

    rhs = T(k)*w_ft

    lhs = 1j*(2*np.pi*kx)*theta+lamda*R(k)

    ker = np.exp(-lhs*t)

    S = ifft2((1/Nt)*fftconvolve(ker,rhs,mode='full',axes=0)).real[0:Nt,:,:]

    return S

def forward_beta(beta):
    # forward operator for slipperiness beta
    # returns the data (elevation) h

    beta_ft = fft2(beta)

    rhs = F(k,kx)*beta_ft

    lhs = 1j*(2*np.pi*kx)*theta+lamda*R(k)

    ker = np.exp(-lhs*t)

    S = ifft2((1/Nt)*fftconvolve(ker,rhs,mode='full',axes=0)).real[0:Nt,:,:]

    return S

def forward_m(m):
    # forward operator for melt rate m
    # returns the data (elevation) h

    m_ft = fft2(m)

    mu = np.sqrt(4*delta*(lamda*B(k))**2 + ((delta-1)**2)*(lamda*Rf(k))**2)

    ker0 = (delta*lamda*B(k)/mu)*np.exp((-1j*(2*np.pi*kx)*theta-lamda*0.5*(delta+1)*Rf(k)+0.5*mu)*t)
    ker1 = (delta*lamda*B(k)/mu)*np.exp((-1j*(2*np.pi*kx)*theta-lamda*0.5*(delta+1)*Rf(k)-0.5*mu)*t)

    S = ifft2((1/Nt)*fftconvolve(-m_ft,ker0-ker1,mode='full',axes=0)).real[0:Nt,:,:]

    return S

def adjoint_m(m):
    # adjoint of the melt rate forward operator

    m_ft = fft2(m)

    mu = np.sqrt(4*delta*(lamda*B(k))**2 + ((delta-1)**2)*(lamda*Rf(k))**2)

    ker0 = (delta*lamda*B(k)/mu)*np.exp((-1j*(2*np.pi*kx)*theta-lamda*0.5*(delta+1)*Rf(k)+0.5*mu)*t)
    ker1 = (delta*lamda*B(k)/mu)*np.exp((-1j*(2*np.pi*kx)*theta-lamda*0.5*(delta+1)*Rf(k)-0.5*mu)*t)

    ker = np.conjugate(ker0-ker1)

    S = ifft2((1/Nt)*fftconvolve(np.flipud(ker),-m_ft,mode='full',axes=0)).real[(Nt-1):2*Nt,:,:]

    return S

def forward_u(w,beta):
    w_ft = fft2(w)
    beta_ft = fft2(beta)
    h = forward_w(w) + forward_beta(beta)
    h_ft = fft2(h)

    F = 0*xi*Ub(k,kx)*beta_ft+1j*(2*np.pi*kx)*(lamda*Uh(k)*h_ft - Uw(k)*w_ft)

    S = ifft2(F).real

    return S

def forward_v(w,beta):
    w_ft = fft2(w)
    beta_ft = fft2(beta)
    h = forward_w(w) + forward_beta(beta)
    h_ft = fft2(h)

    F = 0*xi*Vb(k,kx,ky)*beta_ft+1j*(2*np.pi*ky)*(lamda*Vh(k)*h_ft - Vw(k)*w_ft)

    S = ifft2(F).real

    return S

def adjoint_w(f):
    # adjoint of the basal vertical velocity forward operator

    f_ft = fft2(f)

    lhs = -1j*(2*np.pi*kx)*theta+lamda*R(k)

    ker = T(k)*np.exp(-lhs*t)

    S = ifft2((1/Nt)*fftconvolve(np.flipud(ker),f_ft,mode='full',axes=0)).real[(Nt-1):2*Nt,:,:]

    return S

def adjoint_beta(f):
    # adjoint of the slipperiness forward operator

    f_ft = fft2(f)

    lhs = -1j*(2*np.pi*kx)*theta+lamda*R(k)

    ker = np.conjugate(F(k,kx))*np.exp(-lhs*t)

    S = ifft2((1/Nt)*fftconvolve(np.flipud(ker),f_ft,mode='full',axes=0)).real[(Nt-1):2*Nt,:,:]

    return S


def adj_fwd(X):
    # operator on the LHS of the normal equations:
    # apply forward operator then adjoint operator, and add the regularization term
    if inv_w == 1:
        A = adjoint_w(forward_w(X)) + eps_w*reg(X,w_reg)
    elif inv_beta == 1:
        A = adjoint_beta(forward_beta(X)) + eps_beta*reg(X,beta_reg)
    elif inv_m == 1:
        A = adjoint_m(forward_m(X)) + eps_m*reg(X,m_reg)

    return A
