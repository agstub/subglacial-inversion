# this file contains the integral kernel functions that are used for applying the
# forward and adjoint operators
import numpy as np
from params import beta0,slope,k,kx,ky,uh0,ub0,tau,nu,lamda,t,t_final,Nt
from scipy.signal import fftconvolve

#---------------------convolution and cross-correlation operators---------------
def conv(a,b):
    return (t_final/Nt)*fftconvolve(a,b,mode='full',axes=0)[0:Nt,:,:]

def xcor(a,b):
    return (t_final/Nt)*fftconvolve(np.conjugate(np.flipud(a)),b,mode='full',axes=0)[(Nt-1):2*Nt,:,:]

#------------------------Functions relevant to kernels--------------------------
def Rg():
    # Ice surface relaxation function for grounded ice
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    g = beta0/n
    c_a = (1j*kx/k)*np.tan(slope)
    R1 =  (1/n)*((1+g)*np.exp(4*n) - (2+4*g*n-4*c_a*n*(1+g*n))*np.exp(2*n) + 1 -g)
    D = (1+g)*np.exp(4*n) + (2*g+4*n+4*g*(n**2))*np.exp(2*n) -1 + g

    return R1/D

def Tw():
    # Basal velocity transfer function
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    g = beta0/n
    D = (1+g)*np.exp(4*n) + (2*g+4*n+4*g*(n**2))*np.exp(2*n) -1 + g
    T1 = 2*(1+g)*(n+1)*np.exp(3*n) + 2*(1-g)*(n-1)*np.exp(n)

    return T1/D

def Tb():
    # Basal sliding transfer function
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    nx = 2*np.pi*kx
    g = beta0/n
    F1 =  (2/n)*(np.exp(3*n) + np.exp(n))
    D = (1+g)*np.exp(4*n) + (2*g+4*n+4*g*(n**2))*np.exp(2*n) -1 + g

    return F1/D


def Uw():
    # Horizontal velocity w-response function
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    g = beta0/n

    N = 2*g**2 - 3*g + 2*(2*g**2-1)*np.exp(2*n)+(2*g**2 + 3*g + 1)*np.exp(4*n)+1

    D = ((2*g**2+3*g+1)*np.exp(6*n) + (6*g**2+4*g*(n**2)*(2*g+1)+4*n*(2*g+1)+3*g-1 )*np.exp(4*n)\
    + (6*g**2 + 4*g*(n**2)*(2*g-1) +4*n*(2*g-1)-3*g-1)*np.exp(2*n)+2*g**2-3*g+1)/(2*np.exp(n))

    return N/D

def Uh():
    # Horizontal velocity h-response function
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    nx = 2*np.pi*kx
    c_a = (1j*nx/n)*np.tan(slope)
    g = beta0/n

    N = 2*n*(g*n+1)*(2*g+(2*g+1)*np.exp(2*n)-1)*np.exp(n)

    D = ((2*g**2+3*g+1)*np.exp(6*n) + (6*g**2+4*g*(n**2)*(2*g+1)+4*n*(2*g+1)+3*g-1 )*np.exp(4*n)\
    + (6*g**2 + 4*g*(n**2)*(2*g-1) +4*n*(2*g-1)-3*g-1)*np.exp(2*n)+2*g**2-3*g+1)/(2*np.exp(n))

    kap = (n/nx)**2

    return (1/n**2)*(N+P(n,g,kap,c_a))/D


def Ub():
    # Horizontal velocity beta-response function
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    nx = 2*np.pi*kx
    kap = (n/nx)**2
    g = beta0/n

    N = 2*kap*(g-1)+n*(2*g-1) + 2*(2*g*kap+4*g*n**2*(kap-1)+n*(4*kap-3))*np.exp(2*n)\
        +(2*kap*(g+1)-n*(2*g+1)-1)*np.exp(4*n)+1

    D = ((2*g**2+3*g+1)*np.exp(6*n) + (6*g**2+4*g*(n**2)*(2*g+1)+4*n*(2*g+1)+3*g-1 )*np.exp(4*n)\
    + (6*g**2 + 4*g*(n**2)*(2*g-1) +4*n*(2*g-1)-3*g-1)*np.exp(2*n)+2*g**2-3*g+1)/(2*np.exp(n))

    return (nx**2/n**3)*N/D


def Vw():
    return Uw()


def Vh():
    # Horizontal velocity h-response function
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    nx = 2*np.pi*kx
    g = beta0/n
    c_a = (1j*nx/n)*np.tan(slope)

    N = 2*n*(g*n+1)*(2*g+(2*g+1)*np.exp(2*n)-1)*np.exp(n)

    D = ((2*g**2+3*g+1)*np.exp(6*n) + (6*g**2+4*g*(n**2)*(2*g+1)+4*n*(2*g+1)+3*g-1 )*np.exp(4*n)\
    + (6*g**2 + 4*g*(n**2)*(2*g-1) +4*n*(2*g-1)-3*g-1)*np.exp(2*n)+2*g**2-3*g+1)/(2*np.exp(n))

    return (1/n**2)*(N+P(n,g,0,c_a))/D

def Vb():
    n = 2*np.pi*k           # used to convert to SciPy's Fourier Transform definition
    nx = 2*np.pi*kx
    ny = 2*np.pi*ky
    kap = 0
    g = beta0/n

    N = 2*kap*(g-1)+n*(2*g-1) + 2*(2*g*kap+4*g*n**2*(kap-1)+n*(4*kap-3))*np.exp(2*n)\
        +(2*kap*(g+1)-n*(2*g+1)-1)*np.exp(4*n)+1

    D = ((2*g**2+3*g+1)*np.exp(6*n) + (6*g**2+4*g*(n**2)*(2*g+1)+4*n*(2*g+1)+3*g-1 )*np.exp(4*n)\
    + (6*g**2 + 4*g*(n**2)*(2*g-1) +4*n*(2*g-1)-3*g-1)*np.exp(2*n)+2*g**2-3*g+1)/(2*np.exp(n))

    return (nx*ny/n**3)*N/D

def P(n,g,kap,c_a):
    # Additional terms in velocity response functions for sloping bed problem
    p0 = -1-2*g**2 + 3*g + 2*kap*(2*g**2-3*g+1)
    p1 = (2*g**2 +3*g -2*kap*(2*g**2+3*g+1)+1)*np.exp(6*n)
    p2 = (-16*(g*n)**2 -8*n*g**2 -2*g**2 + 8*g*n**2 -12*g*n -3*g +2*kap*(
           8*(g*n)**2 + 2*g**2 -4*g*n**2 +8*g*n -g-4*n+1)+8*n-1)*np.exp(2*n)
    p3 = (16*(g*n)**2 - 8*n*g**2 +2*g**2 +8*g*n**2 +12*g*n -3*g -2*kap*(
           8*(g*n)**2 +2*g**2 +4*g*n**2 +8*g*n +g+4*n+1)+8*n+1)*np.exp(4*n)

    return c_a*(p0+p1+p2+p3)

Rg_ = Rg()
Tw_ = Tw()
Tb_ = Tb()
Uw_ = Uw()
Uh_ = Uh()
Ub_ = Ub()
Vw_ = Vw()
Vh_ = Vh()
Vb_ = Vb()

#------------------------------ Kernels-----------------------------------------
def ker_w():
    # kernel for w forward problem
    K_h = np.exp(-(1j*(2*np.pi*kx)*uh0+lamda*Rg_)*t)
    K_s = np.exp(-1j*(2*np.pi*kx)*ub0*t)

    K = K_h*Tw_ + 1j*(2*np.pi*kx)*Tb_*tau*conv(K_h,K_s)

    return K

def ker_beta():
    # kernel for beta forward problem
    K_h = np.exp(-(1j*(2*np.pi*kx)*uh0+lamda*Rg_)*t)
    K =  1j*(2*np.pi*kx)*nu*Tb_*K_h
    return K

ker_w_ = ker_w()
ker_beta_ = ker_beta()
ker_s_ = np.exp(-1j*(2*np.pi*kx)*ub0*t)
