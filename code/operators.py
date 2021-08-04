# this file defines the
# * (1) forward operators,
# * (2) adjoint operators, and
# * (3) the operator appearing on the left-side of the normal equations
#       (composition of forward and adjoint operators plus regularization terms)

import numpy as np
from kernel_fcns import R,T,F,Rf,B,Uw,Uh,Ub,Vw,Vh,Vb,Uhf,Usf
from params import (theta,lamda,eps_beta,eps_w,eps_m,w_reg,beta_reg,m_reg,t,k,kx,ky,
                    inv_w,inv_beta,inv_m,delta,Nt,xi,dim,vel_wt,vel_data,k_min)
from scipy.signal import fftconvolve
from scipy.fft import ifft2,fft2
from regularizations import reg

def Hc(X):
    return forward_w(X[0])+forward_beta(X[1])

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

    F = -xi*Ub(k,kx)*beta_ft-1j*(2*np.pi*kx)*(lamda*Uh(k)*h_ft + Uw(k)*w_ft)

    S = ifft2(F).real

    return S

def forward_v(w,beta):
    w_ft = fft2(w)
    beta_ft = fft2(beta)
    h = forward_w(w) + forward_beta(beta)
    h_ft = fft2(h)

    F = -xi*Vb(k,kx,ky)*beta_ft-1j*(2*np.pi*ky)*(lamda*Vh(k)*h_ft + Vw(k)*w_ft)

    S = ifft2(F).real

    return S

def adjoint_Uw(f):
    f_ft = fft2(f)
    p1 = ifft2(1j*(2*np.pi*kx)*(Uw(k)*f_ft)).real
    p2 = adjoint_w(ifft2(1j*(2*np.pi*kx)*(lamda*Uh(k)*f_ft)).real)
    return p1+p2

def adjoint_Vw(f):
    f_ft = fft2(f)
    p1 = ifft2(1j*(2*np.pi*ky)*(Vw(k)*f_ft)).real
    p2 = adjoint_w(ifft2(1j*(2*np.pi*ky)*(lamda*Vh(k)*f_ft)).real)
    return p1+p2

def adjoint_Ub(f):
    f_ft = fft2(f)
    p1 = ifft2(-xi*Ub(k,kx)*f_ft).real
    p2 = adjoint_beta(ifft2(1j*(2*np.pi*kx)*(lamda*Uh(k)*f_ft)).real)
    return p1+p2

def adjoint_Vb(f):
    f_ft = fft2(f)
    p1 = ifft2(-xi*Vb(k,kx,ky)*f_ft).real
    p2 = adjoint_beta(ifft2(1j*(2*np.pi*ky)*(lamda*Vh(k)*f_ft)).real)
    return p1+p2

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

    ker = -F(k,kx)*np.exp(-lhs*t)

    S = ifft2((1/Nt)*fftconvolve(np.flipud(ker),f_ft,mode='full',axes=0)).real[(Nt-1):2*Nt,:,:]

    return S


def adj_fwd(X):
    # operator on the LHS of the normal equations:
    # apply forward operator then adjoint operator, and add the regularization term
    if inv_w == 1 and dim == 1:
        A = adjoint_w(forward_w(X)) + eps_w*reg(X,w_reg)
        if vel_data == 1:
            A += vel_wt*(adjoint_Uw(forward_u(X,0*X)) + adjoint_Vw(forward_v(X,0*X)))
    elif inv_beta == 1 and dim == 1:
        A = adjoint_beta(forward_beta(X)) + eps_beta*reg(X,beta_reg)
        if vel_data == 1:
            A += vel_wt*(adjoint_Ub(forward_u(0*X,X)) + adjoint_Vb(forward_v(0*X,X)))
    elif inv_m == 1 and dim == 1:
        A = adjoint_m(forward_m(X)) + eps_m*reg(X,m_reg)
    elif dim == 2:
        # X[0] = w
        # X[1] = beta

        # LHS of w normal equation
        a1 = adjoint_w(Hc(X)) + eps_w*reg(X[0],w_reg)
        a2 = adjoint_Uw(forward_u(X[0],X[1]))
        a3 = adjoint_Vw(forward_v(X[0],X[1]))
        a = a1+vel_wt*(a2+a3)

        # LHS of beta normal equation
        b1 = adjoint_beta(Hc(X)) + eps_beta*reg(X[1],beta_reg)
        b2 = adjoint_Ub(forward_u(X[0],X[1]))
        b3 = adjoint_Vb(forward_v(X[0],X[1]))
        b = b1+vel_wt*(b2+b3)

        A = np.array([a,b])


    return A




#-------------------------------------------------------------------------------
def forward_s(m):
    # forward operator for melt rate m
    # returns the lower surface elevation s

    m_ft = fft2(m)

    mu = np.sqrt(4*delta*(lamda*B(k))**2 + ((delta-1)**2)*(lamda*Rf(k))**2)

    chi = -(1-delta)*lamda*Rf(k)

    Lp = (-1j*(2*np.pi*kx)*theta-lamda*0.5*(delta+1)*Rf(k)+0.5*mu)*t
    Lm = (-1j*(2*np.pi*kx)*theta-lamda*0.5*(delta+1)*Rf(k)-0.5*mu)*t

    ker0 = (1/(2*mu))*(mu+chi)*np.exp(Lm)
    ker1 = (1/(2*mu))*(mu-chi)*np.exp(Lp)

    S = ifft2((1/Nt)*fftconvolve(m_ft,ker0+ker1,mode='full',axes=0)).real[0:Nt,:,:]

    return S


def forward_uf(h,s):
    h_ft = fft2(h)
    s_ft = fft2(s)

    F = 1j*(2*np.pi*kx)*(Uhf(k)*h_ft + Usf(k)*delta*s_ft)*lamda

    #F[np.abs(k)<k_min] = 0#-lamda*h_ft[np.abs(k)<k_min]

    S = ifft2(F).real

    return S

def forward_vf(h,s):
    h_ft = fft2(h)
    s_ft = fft2(s)

    F = 1j*(2*np.pi*ky)*(Uhf(k)*h_ft + Usf(k)*delta*s_ft)*lamda

#    F[np.abs(k)<k_min] = 0#-lamda*h_ft[np.abs(k)<k_min]

    S = ifft2(F).real

    return S
