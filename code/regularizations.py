# this file contains the regularization options
# * these are just standard Tikhonov regularizations of either
# (1) the function ("L2"), or
# (2) gradients of the functions ("H1")


from params import k,x,y
from scipy.fft import ifft2,fft2

def lap(f):
    # negative Laplacian computed via Fourier transform
    return ifft2((k**2)*fft2(f)).real


def reg(f,reg_type):
    # first variation of regularization functional
    if reg_type == 'H1':
        R = lap(f)
    elif reg_type == 'L2':
        R = f
    return R
