# this file contains a conjugate gradient method implementation that is used to solve
# the normal equations that arise from the least-squares minimization problem

from operators import adj_fwd
from params import dx,dy,dt,cg_tol,max_cg_iter
from scipy.integrate import trapz
import numpy as np

# ------------ define inner products and norms ---------------------------------

def prod(a,b):
    # inner product for the optimization problem: L^2(0,T;L^2) space-time inner product
    int = a*b
    p = trapz( trapz(trapz(int,dx=dx,axis=-2),dx=dy,axis=-1) ,dx=dt,axis=0)

    return p


def norm(a):
    # norm for the optimization problem
    return np.sqrt(prod(a,a))

#------------------------------------------------------------------------------

def cg_solve(b,X0):
# conjugate gradient method for solving the normal equations
#
#              adj_fwd(X) = b,           where...
#
# * adj_fwd is a linear operator defined in operators.py
# * b = right-side vector
# * X0 = initial guess (default zero)

    r0 = b - adj_fwd(X0)      # initial residual
    p = r0                    # initial search direction
    j = 1                     # iteration
    r = r0                    # initialize residual
    X = X0                    # initial guess

    rnorm0 = prod(r,r)    # (squared) norm of the residual: current iteration
    rnorm1 = rnorm0       # (squared) norm of the residual: previous iteration

    while norm(r) > cg_tol:
        if j%20 == 0:
            print("CG iter. "+str(j)+': residual norm = '+"{:.2e}".format(norm(r))+',  tol = '+"{:.2e}".format(cg_tol))

        rnorm0 = prod(r,r)

        Ap = adj_fwd(p)

        alpha = rnorm0/prod(p,Ap)

        X = X + alpha*p                     # update solution
        r = r - alpha*Ap                    # update residual
        rnorm1 = prod(r,r)
        beta0 = rnorm1/rnorm0
        p = r + beta0*p                     # update search direction
        j = j+1

        if j > max_cg_iter:
            print('\n...CG terminated because maximum iterations reached.')
            break
    if j<max_cg_iter:
        print('\n...CG converged!')
    return X
