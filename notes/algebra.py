# this program use SymPy to symbolically derive the relaxation and transfer functions
# see appendix of notes file for a description of the math
import sympy as sp
nu = sp.Symbol('nu')
mu = sp.exp(nu)
gamma = sp.Symbol('gamma')
#gamma = 0 # set friction to zero for ice shelf problem

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

# velocity at upper surface of ice sheet
w_h = mu*sol[0] + (1/mu)*sol[1] + nu*mu*sol[2] + (nu/mu)*sol[3]

# print the result (modulo a 1/k factor) for grounded ice:
sp.pprint(sp.collect( sp.collect(sp.collect(sp.collect(sp.simplify(w_h),b1),b2),b3),mu) )

# print the result (modulo a 1/k factor) for floating ice:
#sp.pprint(sp.collect(sp.collect(sp.collect(sp.simplify(w_h),b1),b3),mu) )

# Also need to print w_b for floating ice, since it is part of the solution
# (modulo a 1/k factor)
#w_b = sol[0]+sol[1]
#sp.pprint(sp.collect(sp.collect(sp.collect(sp.simplify(w_b),b1),b3),mu) )
