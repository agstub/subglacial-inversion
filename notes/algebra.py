# this program use SymPy to symbolically derive the relaxation and transfer functions
# see appendix of notes file for a description of the math
import sympy as sp
nu = sp.Symbol('nu')
mu = sp.exp(nu)
#gamma = sp.Symbol('gamma')
gamma = 0 # set background friction to zero for ice shelf or ice stream problems

# matrix for grounded ice
M = sp.Matrix(( [mu, -1/mu, nu*mu,-nu/mu], [mu, 1/mu, mu*(nu+1),(nu-1)/mu], [1-gamma, 1+gamma, 1-gamma,-1-gamma],[1,1,0,0] ))

# use this matrix for floating ice:
#M = sp.Matrix(( [mu, -1/mu, nu*mu,-nu/mu], [mu, 1/mu, mu*(nu+1),(nu-1)/mu], [1-gamma, 1+gamma, 1-gamma,-1-gamma],[1,-1,0,0] ))


b1 = sp.Symbol('b_1')                   # 1st rhs vector entry: proportional to h
b2 = sp.Symbol('b_2')                   # 2nd rhs vector entry: proportional to beta (only for grounded ice)
b3 = sp.Symbol('b_3')                   # 3rd rhs vector entry: proportial to w_b

# solution vector
A,B,C,D = sp.symbols('A,B,C,D')

# rhs vector for grounded ice:
b = sp.Matrix(4,1,[b1,0,b2,b3])

# rhs vector for floating ice:
#b = sp.Matrix(4,1,[b1,0,0,b3])

sol, = sp.linsolve((M,b),[A,B,C,D])

# vertical velocity at upper surface of ice sheet
w_h = mu*sol[0] + (1/mu)*sol[1] + nu*mu*sol[2] + (nu/mu)*sol[3]

# print the result (modulo a 1/k factor) for grounded ice:
#sp.pprint(sp.collect( sp.collect(sp.collect(sp.collect(sp.simplify(w_h),b1),b2),b3),mu) )

# print the result (modulo a 1/k factor) for floating ice:
#sp.pprint(sp.collect(sp.collect(sp.collect(sp.simplify(w_h),b1),b3),mu) )

# Also need to print w_b for floating ice, since it is part of the solution
# (modulo a 1/k factor)
#w_b = sol[0]+sol[1]
#sp.pprint(sp.collect(sp.collect(sp.collect(sp.simplify(w_b),b1),b3),mu) )


# For horizontal surface velocity solutions
# matrix for grounded ice
M2 = sp.Matrix(( [mu, -1/mu],[1-2*gamma, -(1+2*gamma)]))

theta = sp.Symbol('theta')


b4 = -2*(b3 + sol[2]*mu - sol[3]/mu)                   # 1st rhs vector entry
b5 = -2*(-theta*b2 + b3 + sol[2] - sol[3])                              # 2nd rhs vector entry

# solution vector
E,F = sp.symbols('E,F')


# rhs vector for grounded ice:
d = sp.Matrix(2,1,[b4,b5])

sol2, = sp.linsolve((M2,d),[E,F])

uh0 = sol2[0]*mu + sol2[1]/mu
uh1 = K*(sol[0]*(mu-1) + sol[1]*(1-1/mu) + sol[2]*(mu*(1+nu)-1) + sol[3]*((1-nu)/mu - 1))

uh = uh0 + uh1

## print velocity response functions
#sp.pprint( sp.simplify(sp.collect( sp.collect(sp.collect(sp.collect(sp.simplify(uh),b1),b2),b3),mu) ))

#sp.pprint(sp.simplify(sp.collect(sp.collect(sp.simplify(uh),b1),mu)) )
