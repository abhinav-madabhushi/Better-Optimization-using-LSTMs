#!/usr/bin/env python3
import numpy as np
import os
import aesara
aesara.config.linker = 'vm'          # use the Python VM backend
aesara.config.optimizer = 'fast_compile'  
import aesara.tensor as at

# adjust these imports if your files live in a subpackage
from problems.Convex_QCQP import Convex_QCQP
from pdProj import pdProj

def load_qcqp_instance(mat_path, idx=0):
    """
    Load the idx-th problem from a .mat via Convex_QCQP,
    return (Q, p, Q_ineq, G, c, A, b) as NumPy arrays.
    """
    prob = Convex_QCQP(
        prob_type='Convex_QCQP_RHS',
        learning_type='test',
        file_path=mat_path
    )
    
    # grab the idx-th sample
    Q_t      = prob.Q[idx]          # torch.Tensor [n,n]
    p_t      = prob.p[idx,...,0]    # torch.Tensor [n]
    Q_ineq_t = prob.Q_ineq          # torch.Tensor [m1,n,n]
    G_t      = prob.G[idx]          # torch.Tensor [m1,n]
    c_t      = prob.c[idx,...,0]    # torch.Tensor [m1]
    A_t      = prob.A[idx]          # torch.Tensor [m2,n]
    b_t      = prob.b[idx,...,0]    # torch.Tensor [m2]

    return (
        Q_t.cpu().numpy(),
        p_t.cpu().numpy(),
        Q_ineq_t.cpu().numpy(),
        G_t.cpu().numpy(),
        c_t.cpu().numpy(),
        A_t.cpu().numpy(),
        b_t.cpu().numpy(),
    )

def build_aesara_expressions(Q, p, Q_ineq, G, c, A, b):
    x_dev = at.vector('x_dev')

    # objective: 0.5 xᵀQx + pᵀx
    f_expr = 0.5 * at.dot(x_dev, at.dot(at.constant(Q), x_dev)) \
           + at.dot(at.constant(p), x_dev)

    # inequalities: g_i(x) = xᵀQ_ineq[i] x + G[i]ᵀx − c[i]  ≤ 0
    ineqs = []
    for i in range(Q_ineq.shape[0]):
        Qi = at.constant(Q_ineq[i])
        Gi = at.constant(G[i])
        ci = float(c[i])
        ineqs.append(at.dot(x_dev, at.dot(Qi, x_dev))
                    + at.dot(Gi, x_dev)
                    - ci)

    # equalities: h_j(x) = A[j]ᵀ x − b[j] = 0
    eqs = []
    for j in range(A.shape[0]):
        Aj = at.constant(A[j])
        bj = float(b[j])
        eqs.append(at.dot(Aj, x_dev) - bj)

    # stack all constraints  
    c_expr = at.stack(ineqs + eqs)

    return x_dev, f_expr, c_expr

def main():
    # === load one problem instance ===
    mat_file = "random_convex_qcqp_dataset_var100_ineq50_eq50_ex10000"
    file_path = os.path.join('datasets', 'convex_qcqp', f"{mat_file}.mat")
    problem = Convex_QCQP(
        prob_type='Convex_QCQP_RHS',
        learning_type='test',
        file_path=file_path
    )
    B = problem.test_size
    iters_list = []
    for idx in range(B):
        Q, p, Q_ineq, G, c, A, b = load_qcqp_instance(file_path, idx=idx)

        # === build Aesara expressions ===
        x_dev, f_expr, c_expr = build_aesara_expressions(Q, p, Q_ineq, G, c, A, b)

        # === variable bounds ===
        n   = Q.shape[1]
        INF = 1e15
        bl  = np.full(n, -INF)
        bu  = np.full(n,  INF)

        # === constraint bounds ===
        m1 = Q_ineq.shape[0]
        m2 = A.shape[0]
        cl = np.concatenate([np.full(m1, -INF), np.zeros(m2)])
        cu = np.zeros(m1 + m2)

        # === initial guess ===
        if m2 > 0:
            x0 = np.linalg.pinv(A).dot(b)
        else:
            x0 = np.zeros(n)

        # === solve with pdProj ===
        solver = pdProj(
            x0=x0,
            bl=bl, bu=bu,
            cl=cl, cu=cu,
            x_dev=x_dev,
            f=f_expr,
            c=c_expr,
            infinity=INF,
            printLevel=1
        )

        x_sol, y_sol, nvar, ncon, status, iters, nfev, elapsed, fval = solver.solve()
        iters_list.append(iters) 


        # === print results ===
        print("pdProj status      :", status)
        print("Iterations         :", iters)
        print("Function evals     :", nfev)
        print("Elapsed time (s)   :", elapsed)
        print("Objective value    :", fval)
        print("Primal solution x  :", x_sol)
        print("Dual multipliers y :", y_sol)
    iter_final = sum(iters_list) / len(iters_list)
    print(iter_final)


if __name__ == '__main__':
    main()
