# this file contains the integral kernel functions that are used for applying the
# forward and adjoint operators
import numpy as np
from params import k,kx,beta0,U

def R(k):
    # Ice surface relaxation function
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    g = beta0/n
    R1 =  (1/n)*((1+g)*np.exp(4*n) - (2+4*g*n)*np.exp(2*n) + 1 -g)
    D = (1+g)*np.exp(4*n) + (2*g+4*n+4*g*(n**2))*np.exp(2*n) -1 + g

    return R1/D


def T(k):
    # Basal velcoity transfer function
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    g = beta0/n
    D = (1+g)*np.exp(4*n) + (2*g+4*n+4*g*(n**2))*np.exp(2*n) -1 + g
    T1 = 2*(1+g)*(n+1)*np.exp(3*n) + 2*(1-g)*(n-1)*np.exp(n)

    return T1/D

def F(k,kx):
    # Friction transfer function
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    nx = 2*np.pi*kx
    F1 =  U*(2*1j*nx)*(np.exp(3*n) + np.exp(n))
    D = n*(np.exp(4*n) + 4*n*np.exp(2*n) -1)

    return F1/D
