# this file contains a conjugate gradient method implementation that is used to solve
# the normal equations that arise from the least-squares minimization problem

from operators import adj_fwd
from params import dx,dy,dt,cg_tol,max_cg_iter
from scipy.integrate import trapz
import numpy as np


# ------------ define inner products and norms for CG method--------------------

def prod(a,b):
    # inner product for the optimization problem: L^2(0,T;L^2) space-time inner product
    if np.size(np.shape(a)) == 3:
        int = a*b
    elif np.size(np.shape(a)) == 4:
        int = a[0]*b[0] + a[1]*b[1]

    p = trapz( trapz(trapz(int,dx=dx,axis=-2),dx=dy,axis=-1) ,dx=dt,axis=0)

    return p


def norm(a):
    # norm for the optimization problem
    return np.sqrt(prod(a,a))

#------------------------------------------------------------------------------

def cg_solve(b,inv_w,inv_beta,eps_w,eps_beta,vel_locs):
# conjugate gradient method for solving the normal equations
#
#              adj_fwd(X)  = b,           where...
#
# * adj_fwd is a linear operator defined in operators.py
# * b = right-side vector

    r0 = b                    # initial residual
    p = r0                    # initial search direction
    j = 1                     # iteration
    r = r0                    # initialize residual
    X = 0*p                   # initial guess

    rnorm0 = prod(r,r)    # (squared) norm of the residual: previous iteration
    rnorm1 = rnorm0       # (squared) norm of the residual: current iteration

    r00 = norm(b)

    while np.sqrt(rnorm1)/r00 > cg_tol:
        if j%10 == 0:
            print("CG iter. "+str(j)+': rel. residual norm = '+"{:.2e}".format(np.sqrt(rnorm1)/r00)+',  tol = '+"{:.2e}".format(cg_tol))


        rnorm0 = prod(r,r)

        Ap = adj_fwd(p,inv_w,inv_beta,eps_w,eps_beta,vel_locs)

        alpha_c = rnorm0/prod(p,Ap)

        X = X + alpha_c*p                     # update solution
        r = r - alpha_c*Ap                    # update residual
        rnorm1 = prod(r,r)
        beta_c = rnorm1/rnorm0
        p = r + beta_c*p                     # update search direction
        j = j+1


        if j > max_cg_iter:
            print('\n...CG terminated because maximum iterations reached.')
            break
    if j<max_cg_iter:
        print('\n...CG converged!')
    return X
