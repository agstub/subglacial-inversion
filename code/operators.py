# this file defines the
# * (1) forward operators,
# * (2) adjoint operators, and
# * (3) the operator appearing on the left-side of the normal equations
#       (composition of forward and adjoint operators plus regularization terms)

import numpy as np
from kernel_fcns import Rg,Tw,Tb,Rf,B,Uw,Uh,Ub,Vw,Vh,Vb,Uhf,Usf
from params import (uh0,ub0,lamda,eps_beta,eps_w,eps_m,w_reg,beta_reg,m_reg,t,k,kx,ky,
                    inv_w,inv_beta,inv_m,delta,Nt,nu,dim,u_wt,h_wt,tau,vel_data)
from scipy.signal import fftconvolve
from scipy.fft import ifft2,fft2
from regularizations import reg

#---------------------convolution and cross-correlation operators---------------
def conv(a,b):
    return (1/Nt)*fftconvolve(a,b,mode='full',axes=0)[0:Nt,:,:]

def xcor(a,b):
    return (1/Nt)*fftconvolve(np.conjugate(np.flipud(a)),b,mode='full',axes=0)[(Nt-1):2*Nt,:,:]

#-------------------------------------------------------------------------------
def adj_fwd(X):
    # operator on the LHS of the normal equations:
    # apply forward operator then adjoint operator, and add the regularization term
    if inv_w == 1 and dim == 1:
        A = h_wt()*adjoint_w(forward_w(X)) + eps_w()*reg(X,w_reg)
        if vel_data == 1:
            A += u_wt()*(adjoint_Uw(forward_U(X,0*X)) + adjoint_Vw(forward_V(X,0*X)))
    elif inv_beta == 1 and dim == 1:
        A = h_wt()*adjoint_beta(forward_beta(X)) + eps_beta()*reg(X,beta_reg)
        if vel_data == 1:
            A += u_wt()*(adjoint_Ub(forward_U(0*X,X)) + adjoint_Vb(forward_V(0*X,X)))
    elif inv_m == 1 and dim == 1:
        A = adjoint_m(forward_m(X)) + eps_m()*reg(X,m_reg)
    elif dim == 2:
        # X[0] = w
        # X[1] = beta

        # LHS of w normal equation
        a1 = adjoint_w(Hc(X)) + eps_w()*reg(X[0],w_reg)
        a2 = adjoint_Uw(forward_U(X[0],X[1]))
        a3 = adjoint_Vw(forward_V(X[0],X[1]))
        a = h_wt()*a1+u_wt()*(a2+a3)

        # LHS of beta normal equation
        b1 = adjoint_beta(Hc(X)) + eps_beta()*reg(X[1],beta_reg)
        b2 = adjoint_Ub(forward_U(X[0],X[1]))
        b3 = adjoint_Vb(forward_V(X[0],X[1]))
        b = h_wt()*b1+u_wt()*(b2+b3)

        A = np.array([a,b])

    return A

def Hc(X):
    # coupled (w and beta) elevation solution operator
    return forward_w(X[0])+forward_beta(X[1])

#------------------------------ Kernels-----------------------------------------
def ker_w():
    # kernel for w forward problem
    K_h = np.exp(-(1j*(2*np.pi*kx)*uh0+lamda*Rg(k,kx))*t)
    K_s = np.exp(-1j*(2*np.pi*kx)*ub0*t)

    K = K_h*Tw(k) + 1j*(2*np.pi*kx)*Tb(k,kx)*tau*conv(K_h,K_s)

    return K

def ker_beta():
    # kernel for beta forward problem
    K_h = np.exp(-(1j*(2*np.pi*kx)*uh0+lamda*Rg(k,kx))*t)
    K =  1j*(2*np.pi*kx)*nu*Tb(k,kx)*K_h
    return K

def ker_m():
    mu = np.sqrt(4*delta*(lamda*B(k))**2 + ((delta-1)**2)*(lamda*Rf(k))**2)

    ker0 = (delta*lamda*B(k)/mu)*np.exp((-1j*(2*np.pi*kx)*uh0-lamda*0.5*(delta+1)*Rf(k)+0.5*mu)*t)
    ker1 = (delta*lamda*B(k)/mu)*np.exp((-1j*(2*np.pi*kx)*uh0-lamda*0.5*(delta+1)*Rf(k)-0.5*mu)*t)

    K = ker1-ker0
    return K

#---------------------Ice-surface elevation solution operators------------------
def forward_w(w):
    # forward operator for basal vertical velocity w
    # returns the data (elevation) h

    w_ft = fft2(w)

    S = ifft2(conv(ker_w(),w_ft)).real

    return S

def adjoint_w(f):
    # adjoint of the basal vertical velocity forward operator

    f_ft = fft2(f)

    S = ifft2(xcor(ker_w(),f_ft)).real

    return S

def forward_beta(beta):
    # forward operator for slipperiness beta
    # returns the data (elevation) h

    beta_ft = fft2(beta)

    S = ifft2(conv(ker_beta(),beta_ft)).real

    return S

def adjoint_beta(f):
    # adjoint of the beta forward operator

    f_ft = fft2(f)

    S = ifft2(xcor(ker_beta(),f_ft)).real

    return S

def forward_m(m):
    # forward operator for sub-shelf melt rate m
    # returns the data (elevation) h

    m_ft = fft2(m)

    S = ifft2(conv(ker_m(),m_ft)).real

    return S

def adjoint_m(m):
    # adjoint of the melt rate forward operator

    m_ft = fft2(m)

    S = ifft2(xcor(ker_m(),m_ft)).real

    return S

#-----------------------Velocity solution operators-----------------------------
def forward_U(w,beta):
    # u-component for grounded ice
    w_ft = fft2(w)
    beta_ft = fft2(beta)
    h = forward_w(w) + forward_beta(beta)
    h_ft = fft2(h)

    F = -Ub(k,kx)*(nu*beta_ft+tau*sg_fwd(w))-1j*(2*np.pi*kx)*(lamda*Uh(k,kx)*h_ft + Uw(k)*w_ft)

    S = ifft2(F).real

    return S

def forward_V(w,beta):
    # v-component for grounded ice
    w_ft = fft2(w)
    beta_ft = fft2(beta)
    h = forward_w(w) + forward_beta(beta)
    h_ft = fft2(h)

    F = -Vb(k,kx,ky)*(nu*beta_ft+tau*sg_fwd(w))-1j*(2*np.pi*ky)*(lamda*Vh(k,kx)*h_ft + Vw(k)*w_ft)

    S = ifft2(F).real

    return S

def adjoint_Uw(f):
    f_ft = fft2(f)
    p1 = ifft2(1j*(2*np.pi*kx)*(Uw(k)*f_ft)).real
    p2 = adjoint_w(ifft2(1j*(2*np.pi*kx)*(lamda*np.conjugate(Uh(k,kx))*f_ft)).real)
    p3 = ifft2(-Ub(k,kx)*tau*sg_adj(f_ft)).real
    return p1+p2+p3

def adjoint_Vw(f):
    f_ft = fft2(f)
    p1 = ifft2(1j*(2*np.pi*ky)*(Vw(k)*f_ft)).real
    p2 = adjoint_w(ifft2(1j*(2*np.pi*ky)*(lamda*np.conjugate(Vh(k,kx))*f_ft)).real)
    p3 = ifft2(-Vb(k,kx,ky)*tau*sg_adj(f_ft)).real
    return p1+p2+p3

def adjoint_Ub(f):
    f_ft = fft2(f)
    p1 = ifft2(-nu*Ub(k,kx)*f_ft).real
    p2 = adjoint_beta(ifft2(1j*(2*np.pi*kx)*(lamda*np.conjugate(Uh(k,kx))*f_ft)).real)
    return p1+p2

def adjoint_Vb(f):
    f_ft = fft2(f)
    p1 = ifft2(-nu*Vb(k,kx,ky)*f_ft).real
    p2 = adjoint_beta(ifft2(1j*(2*np.pi*ky)*(lamda*np.conjugate(Vh(k,kx))*f_ft)).real)
    return p1+p2

def forward_Uf(h,s):
    # u-component for floating ice
    h_ft = fft2(h)
    s_ft = fft2(s)

    F = 1j*(2*np.pi*kx)*(Uhf(k)*h_ft + Usf(k)*delta*s_ft)*lamda

    S = ifft2(F).real

    return S

def forward_Vf(h,s):
    # v-component for floating ice
    h_ft = fft2(h)
    s_ft = fft2(s)

    F = 1j*(2*np.pi*ky)*(Uhf(k)*h_ft + Usf(k)*delta*s_ft)*lamda

    S = ifft2(F).real

    return S

#---------------------Operators for lower surface elevation---------------------
def sg_fwd(w_ft):
    # forward operator for lower surface elevation
    # returns the fourier-transformed lower surface elevation
    ker = np.exp(-1j*(2*np.pi*kx)*ub0*t)
    S = conv(ker,w_ft)
    return S

def sg_adj(f_ft):
    # adjoint operator for lower surface elevation
    # returns fourier-transformed adjoint
    ker = np.exp(-1j*(2*np.pi*kx)*ub0*t)
    S = xcor(ker,f_ft)
    return S

def forward_s(m):
    # returns the lower surface elevation s

    m_ft = fft2(m)

    mu = np.sqrt(4*delta*(lamda*B(k))**2 + ((delta-1)**2)*(lamda*Rf(k))**2)

    chi = -(1-delta)*lamda*Rf(k)

    Lp = (-1j*(2*np.pi*kx)*uh0-lamda*0.5*(delta+1)*Rf(k)+0.5*mu)*t
    Lm = (-1j*(2*np.pi*kx)*uh0-lamda*0.5*(delta+1)*Rf(k)-0.5*mu)*t

    ker0 = (1/(2*mu))*(mu+chi)*np.exp(Lm)
    ker1 = (1/(2*mu))*(mu-chi)*np.exp(Lp)

    ker = ker0+ker1

    S = ifft2(conv(ker,m_ft)).real

    return S
