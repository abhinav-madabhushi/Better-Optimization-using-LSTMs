import numpy as np
import scipy.io as sio

# Load your dataset
data = sio.loadmat('datasets/qp/unconstrained_QP.mat')
train_size = int(10000 * (1 - 0.0006 - 0.9800))
test_size = int(10000 * (0.9800))
train = True
if (train):
    Q  = data['Q'][:train_size]  # (N,2,2)
    p  = data['p'][:train_size]   # (N,2)
    lb = data['lb'][:train_size] # (N,2) with -inf where no lower bound
    ub = data['ub'][:train_size]  # (N,2) with +inf where no upper bound
else:
    Q  = data['Q'][-test_size:]  # (N,2,2)
    p  = data['p'][-test_size:]   # (N,2)
    lb = data['lb'][-test_size:] # (N,2) with -inf where no lower bound
    ub = data['ub'][-test_size:]  # (N,2) with +inf where no upper bound

N, n = p.shape
x_star = np.zeros((N, n))

for i in range(N):
    Qi = Q[i]
    pi = p[i]
    # Unconstrained optimum
    xi = -np.linalg.solve(Qi, pi)
    # Project onto box [lb, ub]
    xi = np.maximum(lb[i], np.minimum(ub[i], xi))
    x_star[i] = xi

print("Minimum: ", x_star.mean().item())

quad_term = 0.5 * np.einsum('ni,nij,nj->n', x_star, Q, x_star)

# pᵢᵀ xᵢ
lin_term  = (p * x_star).sum(axis=1)

f_vals = quad_term + lin_term          # shape (N,)

mean_f = f_vals.mean()

print("Mean function value f(x*) over the data set: ", mean_f)

# Example: print first 5 solutions
for i in range(5):
    print(f"Problem {i}: x* = {f_vals[i]}")

import numpy as np
import scipy.io as sio

# 1) Load data
data = sio.loadmat('datasets/qp/unconstrained_QP.mat')
#train_size = int(10000 * (1 - 0.0833 - 0.0833))
if (train):
    Qs  = data['Q'][:train_size]  # (N,2,2)
    ps  = data['p'][:train_size]   # (N,2)
    lbs = data['lb'][:train_size] # (N,2) with -inf where no lower bound
    ubs = data['ub'][:train_size]  # (N,2) with +inf where no upper bound
else:
    Qs  = data['Q'][-test_size:]  # (N,2,2)
    ps  = data['p'][-test_size:]   # (N,2)
    lbs = data['lb'][-test_size:] # (N,2) with -inf where no lower bound
    ubs = data['ub'][-test_size:]  # (N,2) with +inf where no upper bound

N, n = ps.shape

# 2) Gradient‐descent parameters
max_iter = 1000
tol      = 1e-8

# To pick a safe step‐size α < 1/L where L=max_eig(Q)
# Since each Q is diagonal, its largest eigenvalue is its max diagonal entry:
Ls = np.max(np.abs(Qs[:,0,0]), axis=0)
Ls = np.maximum(Ls, np.max(np.abs(Qs[:,1,1]), axis=0))
alpha = 0.9 / Ls  # a little under 1/L

# 3) Run projected gradient descent
x_star = np.zeros((N,n))
for i in range(N):
    Q, p = Qs[i], ps[i]
    lb, ub = lbs[i], ubs[i]
    # init at midpoint if both bounds finite, else zero
    x = np.zeros(n)
    finite_lb = np.isfinite(lb)
    finite_ub = np.isfinite(ub)
    both = finite_lb & finite_ub
    x[both] = 0.5*(lb[both] + ub[both])

    for it in range(max_iter):
        grad = Q.dot(x) + p               # ∇f(x) = Q x + p
        x_new = x - alpha*grad            # gradient descent step
        # project onto [lb,ub] componentwise
        x_new = np.minimum(np.maximum(x_new, lb), ub)

        if np.linalg.norm(x_new-x, np.inf) < tol:
            break
        x = x_new

    x_star[i] = x

# 1) quadratic term:  [N] array of x_i^T Q_i x_i
quad = np.einsum('ni,nij,nj->n', x_star, Qs, x_star)

# 2) linear term:   [N] array of p_i^T x_i
lin  = (ps * x_star).sum(axis=1)

obj_vals = 0.5*quad + lin

# print first few
for i in range(5):
    print(f"obj[{i}] = {obj_vals[i]:.6f}")


import numpy as np
import scipy.io as sio
from cyipopt import minimize_ipopt

# 1) Load data
data = sio.loadmat('datasets/qp/unconstrained_QP.mat')
#train_size = int(10000 * (1 - 0.0833 - 0.0833))
if (train):
    Q_data  = data['Q'][:train_size]  # (N,2,2)
    p_data  = data['p'][:train_size]   # (N,2)
    lb_data = data['lb'][:train_size] # (N,2) with -inf where no lower bound
    ub_data = data['ub'][:train_size]  # (N,2) with +inf where no upper bound
else:
    Q_data  = data['Q'][-test_size:]  # (N,2,2)
    p_data  = data['p'][-test_size:]   # (N,2)
    lb_data = data['lb'][-test_size:] # (N,2) with -inf where no lower bound
    ub_data = data['ub'][-test_size:]  # (N,2) with +inf where no upper bound

N, n = p_data.shape
solutions = np.zeros((N, n))
z1_vals   = np.zeros((N, n))        # lower-bound multipliers
z2_vals   = np.zeros((N, n))        # upper-bound multipliers

# 2) Loop over each problem
for i in range(N):
    Q = Q_data[i]
    p = p_data[i]
    lb = lb_data[i]
    ub = ub_data[i]

    # objective and gradient for this instance
    def obj(x, Q=Q, p=p):
        # 0.5 x^T Q x + p^T x
        return 0.5 * x.dot(Q.dot(x)) + p.dot(x)

    def grad(x, Q=Q, p=p):
        # ∇f = Q x + p
        return Q.dot(x) + p

    # variable bounds as list of (lower,upper) pairs
    bounds = [(lb[j], ub[j]) for j in range(n)]

    # initial guess (mid‐point of bounds or zero)
    x0 = np.zeros(n)
    finite_lb = np.isfinite(lb)
    finite_ub = np.isfinite(ub)
    both = finite_lb & finite_ub
    x0[both] = 0.5*(lb[both] + ub[both])

    # 3) call Ipopt
    res = minimize_ipopt(
        fun=obj,
        x0=x0,
        jac=grad,
        bounds=bounds,
        options={
            'tol'       : 1e-8,
            'max_iter'  : 100,
            'print_level': 0
        }
    )
    solutions[i] = res.x
    z1_vals[i]   = res.info["mult_x_L"]   # ← z₁
    z2_vals[i]   = res.info["mult_x_U"]   # ← z₂

# 4) Inspect results
# Compute objective values to verify
quad = np.einsum('ni,nij,nj->n', solutions, Q_data, solutions)
lin  = (p_data * solutions).sum(axis=1)
obj_vals = 0.5*quad + lin

print("Mean function value: ", obj_vals.mean().item())

print("First 5 solutions, objective,  z1,  z2")
for i in range(5):
    print(f"Problem {i}: "
          f"x* = {solutions[i]}, "
          f"f(x*) = {obj_vals[i]:.6f}, "
          f"z1 = {z1_vals[i]}, "
          f"z2 = {z2_vals[i]}")


