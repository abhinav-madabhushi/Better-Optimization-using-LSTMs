#!/usr/bin/env python
import os
import time
import torch
import configargparse
import copy
import matplotlib.pyplot as plt

from problems.Convex_QCQP import Convex_QCQP

def run_newton_on_instance(problem, idx, 
                           tol=1e-2,
                           max_iters=50,
                           sigma=0.1,
                           reg=1e-3,
                           init_alpha=1.0,
                           ls_tau=0.5,
                           ls_c=1e-5, 
                           prob_type = 'Convex_QCQP_RHS', newton_flag = True, device = 'cpu'):
    # helper to slice [idx:idx+1] along dim 0
    prob_copy = copy.deepcopy(problem)
    def one(t):
        return t[idx:idx+1].clone()
    
    print(problem.Q.shape)

    # slice every data tensor
    prob_copy.Q       = one(prob_copy.Q)
    prob_copy.p       = one(prob_copy.p)
    prob_copy.A       = one(prob_copy.A)
    prob_copy.b       = one(prob_copy.b)
    prob_copy.Q_ineq  = prob_copy.Q_ineq  # these are shared across all samples
    prob_copy.G       = one(prob_copy.G)
    prob_copy.c       = one(prob_copy.c)
    #problem.lb       = one(problem.lb)
    #problem.ub       = one(problem.ub)

    # update sizes
    prob_copy.test_size = 1
    batch_size        = 1

    # Use the test split.
    batch_size = prob_copy.test_size
    num_var   = prob_copy.num_var
    num_ineq  = prob_copy.num_ineq
    num_eq    = prob_copy.num_eq
    num_lb = prob_copy.num_lb
    num_ub = prob_copy.num_ub
    # Total KKT system size 
    total_vars = num_var + 2 * num_ineq + num_eq + num_lb + num_ub

    # Initialize variables: x, η, s, lamba, zl0, and zu0. 

        # === Primal variables: strictly feasible by solving A x = b ===
    if num_eq > 0:
        # problem.A: [batch_size, m, n], problem.b: [batch_size, m, 1]
        # x0 = pinv(A) @ b  => shape [batch_size, n, 1]
        A_batch = prob_copy.A                           # [B, m, n]
        b_batch = prob_copy.b                           # [B, m, 1]
        A_pinv   = torch.linalg.pinv(A_batch)        # [B, n, m]
        x0       = torch.bmm(A_pinv, b_batch)        # [B, n, 1]
    else:
        # no equalities ⇒ start at zero
        x0 = torch.zeros((batch_size, num_var, 1), device=device)

    # === Inequality slacks & duals: enforce g(x0)+s0 = 0 strictly ===
    if num_ineq > 0:
        g0  = prob_copy.ineq_resid(x0)                 # [B, m, 1]
        s0  = torch.clamp(-g0, min=1e-6)             # ensures g0 + s0 = 0 and s0>0
        eta0 = torch.ones_like(s0)                   # positive duals
    else:
        eta0 = torch.zeros((batch_size, 0, 1), device=device)
        s0   = torch.zeros((batch_size, 0, 1), device=device)

    # === Box constraints (unchanged if you have none) ===
    if num_lb > 0:
        x0 = torch.max(x0, prob_copy.lb + 1e-3)
        dl0 = x0 - prob_copy.lb
        zl0 = torch.ones_like(dl0)
    else:
        dl0 = torch.zeros((batch_size, 0, 1), device=device)
        zl0 = torch.zeros((batch_size, 0, 1), device=device)

    if num_ub > 0:
        x0 = torch.min(x0, prob_copy.ub - 1e-3)
        du0 = prob_copy.ub - x0
        zu0 = torch.ones_like(du0)
    else:
        du0 = torch.zeros((batch_size, 0, 1), device=device)
        zu0 = torch.zeros((batch_size, 0, 1), device=device)

    # === Equality multipliers if any (warm‑start at zero) ===
    lamb0 = torch.zeros((batch_size, num_eq, 1), device=device)

    # === Pack into big vector y ===
    y = torch.cat([x0, eta0, s0, lamb0, zl0, zu0], dim=1)

    B       = prob_copy.test_size
    n       = prob_copy.num_var
    m_ineq  = prob_copy.num_ineq
    m_eq    = prob_copy.num_eq
    m_lb    = prob_copy.num_lb
    m_ub    = prob_copy.num_ub
    m       = m_ineq + m_lb + m_ub


    def compute_alpha_pos(z, dz, tau=0.995):
        """
        Compute the maximum step size α to ensure z + α dz > 0
        where dz < 0.
        """
        # Prevent division by zero
        eps = 1e-16  
        ratio = torch.where(dz < 0, -z / (dz + eps), torch.full_like(z, float('inf')))
        return tau * ratio.min()
    

    print("Starting Damped Newton's Method with Regularization for problem type:", prob_type)
    print("Batch size:", batch_size)
    print("Total KKT system size (variables):", total_vars)
    print("Tolerance:", tol)
    print("Max iterations:", max_iters)
    print("Initial α:", init_alpha, "Regularization λ:", reg)
    print("----------------------------------------------------")

    lambda_hessian = 1
    mu_k   = (eta0 * s0).mean().item()    # μ₀ = average η₀·s₀
    eps_mu = mu_k             # inner‐solve tolerance
    sigma  = sigma                   # your centering parameter

    lambda_reg   = reg           # start value
    lambda_min   = 1e-8
    lambda_max   = 1e6
    solve_growth = 10.0               # factor when solve fails
    backoff_fac  = 0.3                # factor to shrink on good progress
    progress_tol = 0.7                # "good progress" threshold
    start_time = time.time()

    total_iters = 0
    outer = 0
    offset = 0

    x    = y[:, offset:offset + num_var, :]
    offset += num_var

    eta  = y[:, offset:offset + num_ineq, :]
    offset += num_ineq

    s    = y[:, offset:offset + num_ineq, :]
    offset += num_ineq

    lamb = y[:, offset:offset + num_eq, :]
    offset += num_eq

    zl   = y[:, offset:offset + num_lb, :]
    offset += num_lb

    zu   = y[:, offset:, :]  # whatever is left
    J, F, J1 = prob_copy.cal_kkt_newton(x, eta, s, lamb, zl, zu, mu=0.0, sigma=sigma, lambs = lambda_hessian, newton_flag = newton_flag)
    res_norm  = F.norm().item()

    # suppose offsets: x-block = F[0:n], η-block = F[n:n+m_ineq], 
    #                  s‑block = F[n+m_ineq:…], λ-block = …, etc.
    # 1) Stationarity residual
    # F1 = F[:, 0:num_var, :]                       
    # dualInf = F1.abs().max().item()

    # # 2) Primal infeasibility (inequalities + equalities)
    # ineq_res = problem.ineq_resid(x)              # [B, m_ineq, 1]
    # eq_res   = problem.eq_resid(x)                # [B, m_eq, 1]
    # primInf  = torch.cat([ineq_res.abs(), eq_res.abs()], dim=1).max().item()

    # # 3) Complementarity measure
    # comp     = (eta * s).abs()                    # [B, m_ineq, 1]
    # ispDualInf = comp.max().item() / max(primInf, 1e-12)
    nu = 1
    iters_armigo = 0
    while mu_k > 1e-8 and res_norm > tol:
        # if primInf <= args.tol_prim and dualInf <= args.tol_dual:
        #     print(f'Converged at iter {iter_idx}: primInf={primInf}, dualInf={dualInf}')
        #     break
        # elif primInf <= args.tol_prim \
        #     and ispDualInf <= primInf * args.tol_isp \
        #     and alpha_s * torch.norm(dx).item() <= (1 + torch.norm(x).item()) * 1e-12 \
        #     and iter_idx > 0:
        #     print(f'Infeasible stationary at iter {iter_idx}')
        #     break

        print(f"\n=== Barrier loop {outer}, μ = {mu_k:.3e} ===")
        offset = 0

        x    = y[:, offset:offset + num_var, :]
        offset += num_var

        eta  = y[:, offset:offset + num_ineq, :]
        offset += num_ineq

        s    = y[:, offset:offset + num_ineq, :]
        offset += num_ineq

        lamb = y[:, offset:offset + num_eq, :]
        offset += num_eq

        zl   = y[:, offset:offset + num_lb, :]
        offset += num_lb

        zu   = y[:, offset:, :]  # whatever is left
        
        J, F, J1 = prob_copy.cal_kkt_newton(x, eta, s, lamb, zl, zu, mu_k, sigma, lambda_hessian, newton_flag = newton_flag)
        res_norm = F.norm().item()
        iter_idx = -1
        while res_norm > eps_mu:
            iter_idx += 1
            total_iters += 1
            # Unpack y into components.
            offset = 0

            x    = y[:, offset:offset + num_var, :]
            offset += num_var

            eta  = y[:, offset:offset + num_ineq, :]
            offset += num_ineq

            s    = y[:, offset:offset + num_ineq, :]
            offset += num_ineq

            lamb = y[:, offset:offset + num_eq, :]
            offset += num_eq

            zl   = y[:, offset:offset + num_lb, :]
            offset += num_lb

            zu   = y[:, offset:, :]  # whatever is left


            # Compute KKT residual F and Jacobian J using cal_kkt.
            J_aff, F_aff, J1 = prob_copy.cal_kkt_newton(x, eta, s, lamb, zl, zu, mu_k, 0, lambda_hessian, newton_flag = newton_flag)
            delta_aff = torch.linalg.solve(J_aff, -F_aff)

            # print("Shape of J1:", J1.shape)
            # print(J1, "matrix J1, second derivative w.r.t x")

            # import numpy as np
            # if np.allclose(J1, J1.T.conj(), atol=1e-12):
            #     eigvals = np.linalg.eigvalsh(J1)          
            # else:
            #     eigvals = np.linalg.eigvals(J1)         
            #     eigvals = eigvals.real            

            # cond_number = eigvals.max() / eigvals.min()
            # print("Eigenvalues of H:", eigvals)
            # print("Condition number of H:", torch.linalg.cond(J1))    

            # print(J1, "matrix J1, second derivative w.r.t x")
            # is_pd = np.all(eigvals > 1e-12)
            # if (is_pd):
            #     print("The matrix J is positive definite.")
            # else:
            #     print("The matrix J is not positive definite.")



            _off = n
            deta_aff = delta_aff[:, _off:_off+m_ineq, :]; _off += m_ineq
            ds_aff  = delta_aff[:, _off:_off+m_ineq, :]

            # affine step length
            alpha_aff = float(min(
                1.0,
                compute_alpha_pos(s, ds_aff).item(),
                compute_alpha_pos(eta, deta_aff).item()
            ))

            # compute μ_aff
            s_aff   = s   + alpha_aff * ds_aff
            eta_aff = eta + alpha_aff * deta_aff
            mu_aff = ((s_aff * eta_aff).sum(dim=1).squeeze(-1) / m).mean().item()

            # 2) Corrector σ_k
            sigma_k = min(1.0, (mu_aff/mu_k)**3)

            J, F, _ = prob_copy.cal_kkt_newton(x, eta, s, lamb, zl, zu, mu_k, sigma_k, lambda_hessian, newton_flag = newton_flag)

            # I  = torch.eye(total_vars, device=device).unsqueeze(0).repeat(batch_size,1,1)
            # # ==========================================================
            # #   robust linear solve with automatic λ growth
            # # ==========================================================
            # tried = 0
            # while True:
            #     try:
            #         J_reg = J + lambda_reg * I
            #         delta = torch.linalg.solve(J_reg, -F)      # may raise or explode
            #         if torch.isnan(delta).any() or delta.norm() > 1e10:
            #             raise RuntimeError("bad step")
            #         break                                      # success
            #     except RuntimeError:
            #         lambda_reg = min(lambda_reg * solve_growth, lambda_max)
            #         tried += 1
            #         if tried > 8:
            #             print("Could not stabilise the linear system")
            #             quit()
            
            
            delta = torch.linalg.solve(J, -F)

            res_norm = F.norm().item()

            # Diagnostics.
            obj_val = prob_copy.obj_fn(x).mean().item()
            ineq_violation = prob_copy.ineq_dist(x).max().item() if num_ineq > 0 else 0.0
            eq_violation = prob_copy.eq_dist(x).max().item() if num_eq > 0 else 0.0
            obj_grad_norm = prob_copy.obj_grad(x).mean().item()
            ineq_grad_norm = prob_copy.ineq_grad(x).mean().item()
            #AT_lamb = prob_copy.AT_lamb(x, eta, s, lamb, zl, zu, sigma).norm().item()
            print(f"Iter: {iter_idx:3d} | Obj: {obj_val:8.4f} | Obj Grad Norm: {obj_grad_norm:12.4f} | Res Norm: {res_norm:12.4f} |"
                f"Max_Ineq: {ineq_violation:8.4f} | Max_Eq: {eq_violation:8.4f} | Inequality Gradient: {ineq_grad_norm:12.4f} | Lambda: {lamb.mean().item():12.4f}")
            
            with torch.no_grad():
                F_blocks = torch.split(F, [num_var, num_ineq, num_ineq, num_eq, num_lb, num_ub], dim=1)
                names = ['Stationarity', 'Ineq primal', 'Ineq complementarity', 'Eq constraints', 'Lower bound comp', 'Upper bound comp']
                for name, block in zip(names, F_blocks):
                    print(f"{name} Norm: {block.norm().item():.4e}")


            if res_norm < tol:
                print("Convergence achieved at iteration", iter_idx)
                break

            # Use a line search to choose a damping factor α.
            alpha = init_alpha
            #f_x = problem.obj_fn(x).mean()
            #g_x = problem.ineq_resid(x)
            #Ax_b = problem.eq_resid(x) if num_eq > 0 else torch.zeros_like(lamb)
            #dl_res = x - problem.lb
            #du_res = problem.ub - x
            #nu = 1.0
            if (iter_idx != 0):
                res_new_feas = max(ineq_violation, eq_violation)   # after recomputing J,F
                res_old_feas = prev_max_violation                  # save each loop

                # Increase ν when feasibility is not improving
                if iter_idx > 0 and res_new_feas > 0.9 * res_old_feas:
                    nu = nu * 10.0          # grow aggressively
                else:
                    nu = max(nu * 0.3, 1.0)  # or decay a bit when things look good
                prev_max_violation = res_new_feas
            else:
                prev_max_violation = max(ineq_violation, eq_violation)


            # Compute merit and gradient
            phi_val, grad_x, grad_s = prob_copy.merit_phi_and_grads(x, s, mu=mu_k, nu=nu)


            offset = 0
            dx = delta[:, offset:offset + num_var, :]
            offset += num_var

            deta = delta[:, offset:offset + num_ineq, :]
            offset += num_ineq

            ds = delta[:, offset:offset + num_ineq, :]
            offset += num_ineq

            dlamb = delta[:, offset:offset + num_eq, :]
            offset += num_eq

            dzl = delta[:, offset:offset + num_lb, :] if num_lb > 0 else torch.zeros_like(zl)
            offset += num_lb

            dzu = delta[:, offset:, :] if num_ub > 0 else torch.zeros_like(zu)

            alpha_s_max   = compute_alpha_pos(s,  ds)              # for slack
            alpha_eta_max = compute_alpha_pos(eta, deta)           # duals
            alpha_zl_max  = compute_alpha_pos(zl, dzl)  if num_lb else torch.tensor(float('inf'), device=device)
            alpha_zu_max  = compute_alpha_pos(zu, dzu)  if num_ub else torch.tensor(float('inf'), device=device)
            alpha_z_max   = torch.min(torch.stack([alpha_eta_max,
                                                alpha_zl_max,
                                                alpha_zu_max]))




            # Compute directional derivative: Dφ(x_k, s_k; p_x, p_s)
            dir_deriv = (grad_x.transpose(1, 2) @ dx + grad_s.transpose(1, 2) @ ds).squeeze(-1)  # shape: (batch,)
            dir_deriv_scalar = dir_deriv.mean().item() 

            alpha_s = min(1.0, alpha_s_max.item()) 
            success = False
            while alpha_s > 1e-8:
                iters_armigo += 1
                x_trial = x + alpha_s * dx
                s_trial = s + alpha_s * ds

                phi_trial, _, _ = prob_copy.merit_phi_and_grads(x_trial, s_trial,mu=mu_k, nu=nu)   # merit uses (x,s) only
                
                if phi_trial <= phi_val + ls_c * alpha_s * dir_deriv_scalar:
                    success = True
                    lambda_hessian = lambda_hessian / 2
                    break
                else:
                    lambda_hessian = lambda_hessian * 2
                alpha_s *= ls_tau                 # shrink

            if not success:
                #lambda_reg = min(lambda_reg * 5.0, lambda_max)
                #print(f"Search failed – increasing λ to {lambda_reg:.1e} and retrying")
                continue                    # restart the outer loop
            

            alpha_z = min(alpha_s, alpha_z_max.item())   # dual step

            x   += alpha_s * dx
            s   += alpha_s * ds
            eta += alpha_z * deta
            lamb+= alpha_z * dlamb
            zl  += alpha_z * dzl
            zu  += alpha_z * dzu

            # re‑assemble the big vector (keeps the rest of the code unchanged)
            y = torch.cat([x, eta, s, lamb, zl, zu], dim=1)
            #res_old = res_norm
            # *** after updating y  (your code) recompute residual ****
            J, F, J1 = prob_copy.cal_kkt_newton(x, eta, s, lamb, zl, zu, mu_k, sigma_k, lambda_hessian, newton_flag = newton_flag)
            res_norm  = F.norm().item()
            # F1 = F[:, 0:num_var, :]                       
            # dualInf = F1.abs().max().item()

            # # 2) Primal infeasibility (inequalities + equalities)
            # ineq_res = problem.ineq_resid(x)              # [B, m_ineq, 1]
            # eq_res   = problem.eq_resid(x)                # [B, m_eq, 1]
            # primInf  = torch.cat([ineq_res.abs(), eq_res.abs()], dim=1).max().item()

            # # 3) Complementarity measure
            # comp     = (eta * s).abs()                    # [B, m_ineq, 1]
            # ispDualInf = comp.max().item() / max(primInf, 1e-12)
            print("Updated res norm: ", res_norm)

            # if res_new <= progress_tol * res_old:
            #     lambda_reg = max(lambda_reg * backoff_fac, lambda_min)
            # Diagnostics.
        
        #mu_k   *= sigma
        #eps_mu *= sigma
        m = num_ineq + num_lb + num_ub
        dot = (eta * s).sum(dim=1).squeeze(-1)    
        mu_k   = (dot / m).mean().item()    # drop the fixed sigma here
        eps_mu = mu_k         
        
        outer += 1
    print("Armigo Iterations: ", iters_armigo)
    function_evals = total_iters + iters_armigo
    gradient_evals = total_iters*2
    hessian_evals = total_iters
    print("Function Evals: ", function_evals)
    print("Gradient Evals: ", gradient_evals)
    print("Hessian Evals: ", hessian_evals)
    return total_iters, function_evals

import numpy as np
import torch
import cyipopt as ipopt

from problems.Convex_QCQP import Convex_QCQP, convex_ipopt

def run_ipopt_on_instance(problem, idx, res_tol=0.1, max_iter=500):
    subprob = copy.deepcopy(problem)

    # 2) slice out only the idx-th sample in each tensor
    #    (assumes each of these is a torch.Tensor with batch dim = 0)
    subprob.Q      = subprob.Q[idx:idx+1].clone()
    subprob.p      = subprob.p[idx:idx+1].clone()
    subprob.Q_ineq = subprob.Q_ineq.clone()    # Q_ineq usually has no batch dim
    subprob.G      = subprob.G[idx:idx+1].clone()
    subprob.c      = subprob.c[idx:idx+1].clone()
    subprob.A      = subprob.A[idx:idx+1].clone()
    subprob.b      = subprob.b[idx:idx+1].clone()

    # 3) if Convex_QCQP stores test_size, override it
    subprob.test_size = 1
    # delegate to the built‑in opt_solver
    _, _, _, ipopt_iters, f_evals, final_obj = subprob.opt_solve(
       solver_type      = 'ipopt',
       tol         = res_tol
    )


    print(f"IPOPT iters={ipopt_iters},  obj={final_obj},  f‑evals={f_evals}")
    # f"g‑evals={grad_evals},  c‑evals={cons_evals},  jac‑evals={jac_evals},  hess‑evals={hess_evals}")

    return ipopt_iters, f_evals



#!/usr/bin/env python3
import os
import numpy as np
import aesara
aesara.config.linker    = 'vm'
aesara.config.optimizer = 'fast_compile'
import aesara.tensor as at
import configargparse

from problems.Convex_QCQP import Convex_QCQP
from pdProj             import pdProj

def load_qcqp_instance(mat_path, idx=0):
    prob = Convex_QCQP(
        prob_type='Convex_QCQP_RHS',
        learning_type='test',
        file_path=mat_path
    )
    Q_t      = prob.Q[idx]          # [n,n]
    p_t      = prob.p[idx,...,0]    # [n]
    Q_ineq_t = prob.Q_ineq          # [m1,n,n]
    G_t      = prob.G[idx]          # [m1,n]
    c_t      = prob.c[idx,...,0]    # [m1]
    A_t      = prob.A[idx]          # [m2,n]
    b_t      = prob.b[idx,...,0]    # [m2]
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
    f_expr = 0.5 * at.dot(x_dev, at.dot(at.constant(Q), x_dev)) \
           + at.dot(at.constant(p), x_dev)
    ineqs = []
    for i in range(Q_ineq.shape[0]):
        Qi, Gi, ci = at.constant(Q_ineq[i]), at.constant(G[i]), float(c[i])
        ineqs.append(at.dot(x_dev, at.dot(Qi, x_dev))
                    + at.dot(Gi, x_dev) - ci)
    eqs = []
    for j in range(A.shape[0]):
        Aj, bj = at.constant(A[j]), float(b[j])
        eqs.append(at.dot(Aj, x_dev) - bj)
    c_expr = at.stack(ineqs + eqs)
    return x_dev, f_expr, c_expr

def run_pdproj_on_instance(mat_path, idx, print_level = 1):
    Q, p, Q_ineq, G, c, A, b = load_qcqp_instance(mat_path, idx)
    x_dev, f_expr, c_expr   = build_aesara_expressions(Q, p, Q_ineq, G, c, A, b)

    # bounds
    n   = Q.shape[1]
    INF = 1e15
    bl  = np.full(n, -INF)
    bu  = np.full(n,  INF)

    # constraint bounds
    m1 = Q_ineq.shape[0]
    m2 = A.shape[0]
    cl = np.concatenate([np.full(m1, -INF), np.zeros(m2)])
    cu = np.zeros(m1 + m2)

    # initial x0
    if m2 > 0:
        x0 = np.linalg.pinv(A).dot(b)
    else:
        x0 = np.zeros(n)

    # solve
    solver = pdProj(
        x0=x0,
        bl=bl, bu=bu,
        cl=cl, cu=cu,
        x_dev=x_dev,
        f=f_expr,
        c=c_expr,
        infinity=INF,
        printLevel=print_level
    )
    x_sol, y_sol, nvar, ncon, status, iters, nfev, elapsed, fval = solver.solve()
    print("pdProj status      :", status)
    print("Iterations         :", iters)
    print("Function evals     :", nfev)
    print("Elapsed time (s)   :", elapsed)
    print("Objective value    :", fval)
    print("Primal solution x  :", x_sol)
    print("Dual multipliers y :", y_sol)
    return iters, nfev

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def performance_profile_from_file(filename, ns, logplot=False, labels=None):
    """
    Replicates pp.m:
      filename : path to whitespace‑delimited file, header + one row per problem
      ns       : number of solvers (number of numeric columns after the first string col)
      logplot  : if True, plots log(ratio) instead of ratio
      labels   : optional list of length ns with legend labels
    """
    # 1) load with pandas (skipping header, using first column as key)
    df = pd.read_csv(filename, delim_whitespace=True, header=0)
    # assume df has at least ns+1 columns, first is string/problem name
    times = df.iloc[:, 1:ns+1].to_numpy()      # shape = (nP, nS)
    nP, nS = times.shape
    print(f"There are {nP} problems and {nS} solvers.")
    
    # 2) min time per problem
    min_time = np.min(np.abs(times), axis=1)   # shape = (nP,)
    
    # 3) performance ratios
    r = times / min_time[:,None]               # broadcast to (nP,nS)
    if logplot:
        r = np.log(r)
    maxr = np.nanmax(r)
    
    # 4) replace NaN (failures) with 2*maxr
    r = np.where(np.isnan(r), 2*maxr, r)
    
    # 5) sort each column ascending
    r_sorted = np.sort(r, axis=0)              # still (nP, nS)
    
    # 6) build CDF y = [1/nP, 2/nP, …, 1]
    y = np.arange(1, nP+1) / float(nP)
    
    # 7) plot
    colors = ['b','r','g','c','k','m']
    if labels is None:
        labels = [f"S{i+1}" for i in range(nS)]
    plt.figure()
    for j in range(nS):
        plt.step(r_sorted[:,j], y,
                 where='post',
                 color=colors[j % len(colors)],
                 linewidth=1.5,
                 label=labels[j])
    plt.xlabel(r'$\tau$ (performance ratio)')
    plt.ylabel(r'$\rho(\tau)$')
    plt.title('Performance Profile')
    plt.axis([0, 1.1*maxr, 0, 1])
    plt.legend()
    plt.grid(True)
    plt.show()


def performance_profile(times, logplot=False, labels=None):
    """
    Same as above but directly from a NumPy array:
      times    : shape (nP, nS) array of iteration counts (or runtimes).
    """
    times = np.asarray(times)
    nP, nS = times.shape
    min_time = np.min(np.abs(times), axis=1)
    r = times / min_time[:,None]
    if logplot:
        r = np.log(r)
    maxr = np.nanmax(r)
    r = np.where(np.isnan(r), 2*maxr, r)
    r_sorted = np.sort(r, axis=0)
    y = np.arange(1, nP+1) / float(nP)
    
    colors = ['b','r','g','c','k','m']
    if labels is None:
        labels = [f"S{i+1}" for i in range(nS)]
    plt.figure()
    for j in range(nS):
        plt.step(r_sorted[:,j], y,
                 where='post',
                 color=colors[j % len(colors)],
                 linewidth=1.5,
                 label=labels[j])
    plt.xlabel(r'$\tau$ (performance ratio)')
    plt.ylabel(r'$\rho(\tau)$')
    plt.title('Performance Profile')
    plt.axis([0, 1.1*maxr, 0, 1])
    plt.legend()
    plt.grid(True)
    plt.show()



def main():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--prob_type', type=str, required=True)
    parser.add_argument('--device',    type=str, default='cpu')
    parser.add_argument('--data_size', type=int, required=True)
    parser.add_argument('--num_var',   type=int, required=True)
    parser.add_argument('--num_eq',    type=int, required=True)
    parser.add_argument('--num_ineq',  type=int, required=True)
    parser.add_argument('--tol',       type=float, default=1e-1)
    parser.add_argument('--max_iters', type=int,   default=50)
    parser.add_argument('--sigma',     type=float, default=0.1)
    parser.add_argument('--reg',       type=float, default=1e-3)
    parser.add_argument('--init_alpha',type=float, default=1.0)
    parser.add_argument('--ls_tau',    type=float, default=0.5)
    parser.add_argument('--ls_c',      type=float, default=1e-5)
    args = parser.parse_args()

    device = args.device
    # load the full test set
    if args.prob_type == 'Convex_QCQP_RHS':
        fname = f"random_convex_qcqp_dataset_var{args.num_var}_ineq{args.num_ineq}_eq{args.num_eq}_ex{args.data_size}.mat"
        path  = os.path.join('datasets','convex_qcqp', fname)
        problem = Convex_QCQP(
            prob_type=args.prob_type,
            learning_type='test',
            file_path=path
        )
    else:
        raise ValueError("Unsupported problem type for profiling")

    # make sure newton and gradient descent runs on each problem seperately, and then we can plot performance profiles easily
    B = problem.test_size
    iters_list_projected = []
    iters_list_newton = []
    iters_list_gradient = []
    iters_list_ipopt = []

    fcn_evals_list_newton = []
    fcn_evals_list_gradient = []
    fcn_evals_list_projected = []
    fcn_evals_list_ipopt = []
    for idx in range(B): 
        newton_iters, fcn_evals_newton = run_newton_on_instance(
            problem, idx,
            tol       = args.tol,
            max_iters = args.max_iters,
            sigma     = args.sigma,
            reg       = args.reg,
            init_alpha=args.init_alpha,
            ls_tau    = args.ls_tau,
            ls_c      = args.ls_c, 
            prob_type = args.prob_type, 
            device = device
        )
        fcn_evals_list_newton.append(fcn_evals_newton)
        iters_list_newton.append(newton_iters)
        gd_iters, fcn_evals_gradient = run_newton_on_instance(
            problem, idx, 
            tol       = args.tol,
            max_iters = args.max_iters,
            sigma     = args.sigma,
            reg       = args.reg,
            init_alpha=args.init_alpha,
            ls_tau    = args.ls_tau,
            ls_c      = args.ls_c, 
            prob_type = args.prob_type, 
            device = device, newton_flag = False
        )
        fcn_evals_list_gradient.append(fcn_evals_gradient)
        iters_list_gradient.append(gd_iters)
        projected_search_ipm_iters, projected_search_ipm_fcn_evals = run_pdproj_on_instance(path, idx)
        iters_list_projected.append(projected_search_ipm_iters) 
        fcn_evals_list_projected.append(projected_search_ipm_fcn_evals)
        ipopt_iters, ipopt_fcn_vals = run_ipopt_on_instance(problem, idx)
        iters_list_ipopt.append(ipopt_iters)
        fcn_evals_list_ipopt.append(ipopt_fcn_vals)
    iter_final_projected = sum(iters_list_projected) / len(iters_list_projected)
    iter_final_newton = sum(iters_list_newton) / len(iters_list_newton)
    iter_final_gradient = sum(iters_list_gradient) / len(iters_list_gradient)
    iter_final_ipopt = sum(iters_list_ipopt) / len(iters_list_ipopt)

    print("Newtons method iterations: ", iter_final_newton)
    print("Gradient Descent iterations: ", iter_final_gradient)
    print("Projected Search IPM iterations: ", iter_final_projected)
    print("IPOPT iterations: ", iter_final_ipopt)

    fcnevals_final_projected = sum(fcn_evals_list_projected) / len(fcn_evals_list_projected)
    fcnevals_final_newton = sum(fcn_evals_list_newton) / len(fcn_evals_list_newton)
    fcnevals_final_gradient = sum(fcn_evals_list_gradient) / len(fcn_evals_list_gradient)
    fcnevals_final_ipopt = sum(fcn_evals_list_ipopt) / len(fcn_evals_list_ipopt)

    print("Newtons method function evals: ", fcnevals_final_newton)
    print("Gradient Descent function evals: ", fcnevals_final_gradient)
    print("Projected Search IPM function evals: ", fcnevals_final_projected)
    print("IPOPT function evals: ", fcnevals_final_ipopt)

    # --- PERFORMANCE PROFILE ---

    # 1) Stack into shape (n_solvers, n_problems)
    all_iters = np.vstack([
        iters_list_projected,
        iters_list_newton,
        iters_list_gradient, 
        iters_list_ipopt
    ])  # shape = (3, B)

    # 2) Best per problem
    best = all_iters.min(axis=0)            # shape (B,)

    # 3) Performance ratios
    ratios = all_iters / best[np.newaxis, :]
    # Replace any infinities or NaNs (e.g. best=0 or failures) with a large penalty:
    r_max = np.nanmax(ratios[np.isfinite(ratios)]) * 10
    ratios[~np.isfinite(ratios)] = r_max

    # 4) Build τ grid
    taus = np.linspace(1, ratios.max(), 100)

    # 5) Compute profiles ρ_s(τ)
    profiles = np.array([
        [(ratios[s] <= τ).mean() for τ in taus]
        for s in range(all_iters.shape[0])
    ])  # shape = (3, len(taus))

     # instead of the above: call our helper
     # all_iters has shape (3, B), but performance_profile expects (nP, nS)
    performance_profile(
         times=all_iters.T,
         logplot=False,
        labels=['Projected IPM','Newton','Gradient Descent', 'IPOPT']
    )

    # # 6) Plot
    # labels = ['Projected IPM', 'Newton', 'Gradient Descent']
    # for si, label in enumerate(labels):
    #     plt.step(taus, profiles[si], where='post', label=label)

    # plt.xlabel(r'$\tau$ (iteration‐count ratio)')
    # plt.ylabel(r'$\rho(\tau)$')
    # plt.title('Performance Profile by Iterations')
    # plt.xlim(1, taus.max())
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.grid(True)
    # plt.show()
if __name__ == "__main__":
    main()
