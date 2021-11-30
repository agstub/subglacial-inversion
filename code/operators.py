# this file defines the
# * (1) forward operators,
# * (2) adjoint operators, and
# * (3) the operator appearing on the left-side of the normal equations
#       (composition of forward and adjoint operators plus regularization terms)

import numpy as np
from kernel_fcns import Rg_,Tw_,Tb_,Uw_,Uh_,Ub_,Vw_,Vh_,Vb_,ker_w_,ker_beta_,ker_s_,conv,xcor
from params import (uh0,ub0,lamda,eps_beta,eps_w,w_reg,beta_reg,t,k,kx,ky,dx,
                    inv_w,inv_beta,Nt,nu,dim,u_wt,h_wt,tau,vel_data,Nt,t_final)
from gps_stats import gps_locs
from scipy.fft import ifft2,fft2
from regularizations import reg

#-------------------------------------------------------------------------------
def adj_fwd(X):
    # operator on the LHS of the normal equations:
    # apply forward operator then adjoint operator, and add the regularization term
    if inv_w == 1 and dim == 1:
        A = h_wt*adjoint_w(forward_w(X)) + eps_w*reg(X,w_reg)
        if vel_data == 1:
            A += u_wt*(adjoint_Uw(forward_U(X,0*X)) + adjoint_Vw(forward_V(X,0*X)))
    elif inv_beta == 1 and dim == 1:
        A = h_wt*adjoint_beta(forward_beta(X)) + eps_beta*reg(X,beta_reg)
        if vel_data == 1:
            A += u_wt*(adjoint_Ub(forward_U(0*X,X)) + adjoint_Vb(forward_V(0*X,X)))*gps_locs
    elif dim == 2:
        # X[0] = w
        # X[1] = beta

        # LHS of w normal equation
        a1 = adjoint_w(Hc(X))
        a2 = adjoint_Uw(forward_U(X[0],X[1]))
        a3 = adjoint_Vw(forward_V(X[0],X[1]))
        a = h_wt*a1+u_wt*(a2+a3)+ eps_w*reg(X[0],w_reg)

        # LHS of beta normal equation
        b1 = adjoint_beta(Hc(X))
        b2 = adjoint_Ub(forward_U(X[0],X[1]))
        b3 = adjoint_Vb(forward_V(X[0],X[1]))
        b = h_wt*b1+u_wt*(b2+b3)+ eps_beta*reg(X[1],beta_reg)

        A = np.array([a,b])

    return A

def Hc(X):
    # coupled (w and beta) elevation solution operator
    return forward_w(X[0])+forward_beta(X[1])


#---------------------Ice-surface elevation solution operators------------------
def forward_w(w):
    # forward operator for basal vertical velocity w
    # returns the data (elevation) h

    w_ft = fft2(w)

    S_ft = conv(ker_w_,w_ft)

    return S_ft

def adjoint_w(f_ft):
    # adjoint of the basal vertical velocity forward operator

    S = ifft2(xcor(ker_w_,f_ft)).real

    return S

def forward_beta(beta):
    # forward operator for slipperiness beta
    # returns the data (elevation) h

    beta_ft = fft2(beta)

    S_ft = conv(ker_beta_,beta_ft)

    return S_ft

def adjoint_beta(f_ft):
    # adjoint of the beta forward operator

    S = ifft2(xcor(ker_beta_,f_ft)).real

    return S

#-----------------------Velocity solution operators-----------------------------
def forward_U(w,beta):
    # u-component for grounded ice
    w_ft = fft2(w)
    beta_ft = fft2(beta)
    h_ft = forward_w(w) + forward_beta(beta)

    S_ft = -Ub_*(nu*beta_ft+tau*sg_fwd(w))-1j*(2*np.pi*kx)*(lamda*Uh_*h_ft + Uw_*w_ft)

    return S_ft

def forward_V(w,beta):
    # v-component for grounded ice
    w_ft = fft2(w)
    beta_ft = fft2(beta)
    h_ft = forward_w(w) + forward_beta(beta)

    S_ft = -Vb_*(nu*beta_ft+tau*sg_fwd(w))-1j*(2*np.pi*ky)*(lamda*Vh_*h_ft + Vw_*w_ft)

    return S_ft

def adjoint_Uw(f_ft):
    p1 = ifft2(1j*(2*np.pi*kx)*(Uw_*f_ft)).real
    p2 = adjoint_w((1j*(2*np.pi*kx)*(lamda*np.conjugate(Uh_)*f_ft)))
    p3 = ifft2(-Ub_*tau*sg_adj(f_ft)).real
    return p1+p2+p3

def adjoint_Vw(f_ft):
    p1 = ifft2(1j*(2*np.pi*ky)*(Vw_*f_ft)).real
    p2 = adjoint_w((1j*(2*np.pi*ky)*(lamda*np.conjugate(Vh_)*f_ft)))
    p3 = ifft2(-Vb_*tau*sg_adj(f_ft)).real
    return p1+p2+p3

def adjoint_Ub(f_ft):
    p1 = ifft2(-nu*Ub_*f_ft).real
    p2 = adjoint_beta((1j*(2*np.pi*kx)*(lamda*np.conjugate(Uh_)*f_ft)))
    return p1+p2

def adjoint_Vb(f_ft):
    p1 = ifft2(-nu*Vb_*f_ft).real
    p2 = adjoint_beta((1j*(2*np.pi*ky)*(lamda*np.conjugate(Vh_)*f_ft)))
    return p1+p2


#---------------------Operators for lower surface elevation---------------------
def sg_fwd(w):
    # forward operator for lower surface elevation
    # returns the fourier-transformed lower surface elevation
    w_ft = fft2(w)
    S_ft = conv(ker_s_,w_ft)
    return S_ft

def sg_adj(f_ft):
    # adjoint operator for lower surface elevation
    # returns fourier-transformed adjoint
    ker = np.exp(-1j*(2*np.pi*kx)*ub0*t)
    S = xcor(ker_s_,f_ft)
    return S
