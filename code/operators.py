# this file defines the
# * (1) forward operator,
# * (2) adjoint operator, and
# * (2) the operator appearing on the left-side of the normal equations
#       (composition of forward and adjoint operators plus regularization terms)

import numpy as np
from kernel_fcns import R,T,F
from params import U,lamda,eps_beta,eps_w,w_reg,beta_reg,t0,t,k,kx,inv_w,inv_beta,inv_couple
from scipy.integrate import cumtrapz
from scipy.fft import ifft2,fft2
from regularizations import reg

def forward(w,beta):
    # forward operator:
    # returns the data (elevation) h given the velocity w and friction beta

    w_ft = fft2(w)
    beta_ft = fft2(beta)

    rhs = T(k)*w_ft + F(k,kx)*beta_ft

    lhs = 1j*(2*np.pi*kx)*U+lamda*R(k)

    f1 = rhs
    f2 = np.exp(lhs*t)

    int = f1*f2

    sol = np.exp(-lhs*t)*cumtrapz(int,t0,axis=0,initial=0.0)

    return ifft2(sol).real

def adjoint_w(f):
    # adjoint of the forward operator with beta set to zero

    f_ft = fft2(f)

    lhs = -1j*(2*np.pi*kx)*U+lamda*R(k)

    int = f_ft*np.exp(-lhs*t)

    sol = T(k)*np.exp(lhs*t)*np.flipud(cumtrapz(np.flipud(int),t0,axis=0,initial=0.0))

    return ifft2(sol).real

def adjoint_beta(f):
    # adjoint of the forward operator with w set to zero

    f_ft = fft2(f)

    lhs = -1j*(2*np.pi*kx)*U+lamda*R(k)

    int = f_ft*np.exp(-lhs*t)

    sol = np.conjugate(F(k,kx))*np.exp(lhs*t)*np.flipud(cumtrapz(np.flipud(int),t0,axis=0,initial=0.0))

    return ifft2(sol).real


def adj_fwd(X):
    # apply forward operator then adjoint operator, and add the regularization term
    if inv_couple == 1:
        w = X[0]
        beta = X[1]
        Aw = adjoint_w(forward(w,beta))  + eps_w*reg(w,w_reg)
        Ab = adjoint_beta(forward(w,beta)) + eps_beta*reg(beta,beta_reg)
        A = np.array([Aw,Ab])
    elif inv_w == 1:
        w = X
        A = adjoint_w(forward(w,0*w)) + eps_w*reg(w,w_reg)
    elif inv_beta == 1:
        beta = X
        A = adjoint_beta(forward(0*beta,beta)) + eps_beta*reg(beta,beta_reg)
    return A
