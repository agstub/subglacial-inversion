import numpy as np
from params import t_final,L,x,y,t
from scipy.interpolate import griddata

def interp(f):
    Nx_f = 101            # fine Nx
    Ny_f = 101            # fine Ny
    Nt_f = 100            # fine Nt

    t0_f = np.linspace(0,t_final,num=Nt_f) # fine time array
    x0_f = np.linspace(-4*L,4*L,num=Nx_f)  # fine x coordinate array
    y0_f = np.linspace(-4*L,4*L,num=Ny_f)  # fine y coordinate array
    t_f,x_f,y_f = np.meshgrid(t0_f,x0_f,y0_f,indexing='ij')

    points = (t_f,x_f,y_f)

    f_fine = griddata((t.ravel(),x.ravel(),y.ravel()),f.ravel(),points)

    return f_fine
