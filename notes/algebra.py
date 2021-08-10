# this program use SymPy to symbolically derive the relaxation and transfer functions
# see appendix of notes file for a description of the math
import sympy as sp
nu = sp.Symbol('nu')
mu = sp.exp(nu)
gamma = sp.Symbol('gamma')
#gamma = 0 # set background friction to zero for ice shelf problem
# matrix for grounded ice
M = sp.Matrix(( [mu, -1/mu, nu*mu,-nu/mu], [mu, 1/mu, mu*(nu+1),(nu-1)/mu], [1-gamma, 1+gamma, 1-gamma,-1-gamma],[1,1,0,0] ))

# use this matrix for floating ice:
#M = sp.Matrix(( [mu, -1/mu, nu*mu,-nu/mu], [mu, 1/mu, mu*(nu+1),(nu-1)/mu], [1-gamma, 1+gamma, 1-gamma,-1-gamma],[1,-1,0,0] ))

q = sp.Symbol('q')

b1 = sp.Function('b1')(nu)          # proportional to h
b2 = q*b1
b3 = sp.Function('b3')(nu)          # proportional to beta
b4 = sp.Function('b4')(nu)          # proportional to w



# solution vector
A,B,C,D = sp.symbols('A,B,C,D')

# rhs vector for grounded ice:
b = sp.Matrix(4,1,[b1,b2,b3,b4])

# rhs vector for floating ice:
#b = sp.Matrix(4,1,[b1,0,0,b3])

sol, = sp.linsolve((M,b),[A,B,C,D])

# vertical velocity at upper surface of ice sheet
w_h = mu*sol[0] + (1/mu)*sol[1] + nu*mu*sol[2] + (nu/mu)*sol[3]

# print the result (modulo a 1/k factor) for grounded ice:
sp.pprint(sp.collect( sp.collect(sp.collect(sp.collect(sp.collect(sp.simplify(w_h),b1),q),b3),b4),mu) )

# print the result (modulo a 1/k factor) for floating ice:
#sp.pprint(sp.collect(sp.collect(sp.collect(sp.simplify(w_h),b1),b3),mu) )

# Also need to print w_b for floating ice, since it is part of the solution
# (modulo a 1/k factor)
#w_b = sol[0]+sol[1]
#sp.pprint(sp.collect(sp.collect(sp.collect(sp.simplify(w_b),b1),b3),mu) )



#-------------------------------------------------------------------------------

A = sol[0]
B = sol[1]
C = sol[2]
D = sol[3]

k = sp.Symbol('k')

kap = sp.Symbol('kappa')


wb = (A+B)/k
wh = (A*mu + B/mu + nu*mu*C + nu*D/mu)/k # correct

wz0 = A-B+C+D           # correct

wzh = A*mu -B/mu +C*mu +C*nu*mu -D*nu/mu + D/mu # correct

wzzh = A*k*mu + B*k/mu + 2*C*k*mu + C*k*nu*mu + D*k*nu/mu  -2*D*k/mu # correct

wzz0 = k*(A+B+2*C-2*D) # correct

Ph = wzh - wz0*(mu+1/mu)/2- wzz0*(1/k)*(mu-1/mu)/2   # P(H) CORRECT

Pzh = wzzh - wz0*k*(mu-1/mu)/2 - wzz0*(mu+1/mu)/2    # P_z(H) CORRECT

b5 = -(k*wh + Pzh/k-0*2*kap*b2)                               # first rhs vector entry

b6 = -(k*wb -0*2*kap*b3)                               # 2nd rhs vector entry

# For horizontal surface velocity solutions
M2 = sp.Matrix(( [mu, -1/mu],[1-2*gamma, -(1+2*gamma)]))

# solution vector
E,F = sp.symbols('E,F')

# rhs vector for grounded ice:
d = sp.Matrix(2,1,[b5,b6])

sol2, = sp.linsolve((M2,d),[E,F])

uh = Ph + sol2[0]*mu + sol2[1]/mu


## print velocity response functions
#sp.pprint( sp.simplify(sp.collect(sp.collect(sp.collect( sp.collect(sp.collect(sp.collect(sp.simplify(uh ),q),b1),b3),b4),mu),kap)))
#print(sp.latex(sp.simplify(sp.collect(sp.collect(sp.collect( sp.collect(sp.collect(sp.collect(sp.simplify(uh ),q),b1),b3),b4),mu),kap))))
