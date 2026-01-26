#!/usr/bin/env python
#import ctypes
#ctypes.windll.kernel32.SetDllDirectoryW(r"C:\Users\krish\Ipopt-3.14.17-win64-msvs2022-md\Ipopt-3.14.17-win64-msvs2022-md\bin")
#!/usr/bin/env python
import os
import time
import torch
import configargparse

# Import your problem classes.
# For a Convex_QCQP_RHS problem, we use Convex_QCQP;
# for a QP_RHS problem, you would import QP instead.
from problems.Convex_QCQP import Convex_QCQP
from problems.QP import QP

def main():

    parser = configargparse.ArgumentParser()

    parser.add_argument('--tol_prim', type=float, default=1e-4,
                    help='Primal‐feasibility tolerance (→tolPrimFeas)')
    parser.add_argument('--tol_dual', type=float, default=1e-4,
                        help='Dual‐feasibility tolerance (→tolDualFeas)')
    parser.add_argument('--tol_isp',  type=float, default=1e-5,
                        help='Infeasible‐stationary tolerance (→tolIsp)')
    
    parser.add_argument('--config', is_config_file=True, type=str, help='Path to config file')
    parser.add_argument('--prob_type', type=str, required=True,
                        help='Problem type (e.g., Convex_QCQP_RHS or QP_RHS)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')

    # Newton's method parameters
    parser.add_argument('--max_iters', type=int, default=50, help='Max number of Newton iterations')
    parser.add_argument('--tol', type=float, default=1e-1, help='Tolerance on the residual norm for convergence')
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma parameter for the KKT conditions')
    parser.add_argument('--reg', type=float, default=1e-3, help='Regularization parameter (λ) for the Newton system')

    # Damping / line search parameters
    parser.add_argument('--init_alpha', type=float, default=1.0, help='Initial damping factor (α)')
    parser.add_argument('--ls_tau', type=float, default=0.5, help='Backtracking reduction factor for α')
    parser.add_argument('--ls_c', type=float, default=1e-4, help='Armijo constant for line search')
    parser.add_argument('--max_ls_iters', type=int, default=10, help='Max line search iterations per Newton step')

    # Data config
    parser.add_argument('--data_size', type=int, default=1000, help='Number of problem instances (data size)')
    parser.add_argument('--num_var', type=int, required=True, help='Number of decision variables')
    parser.add_argument('--num_eq', type=int, required=True, help='Number of equality constraints')
    parser.add_argument('--num_ineq', type=int, required=True, help='Number of inequality constraints')
    args, _ = parser.parse_known_args()

    device = args.device

    # Load the problem instance based on prob_type.
    if args.prob_type == 'Convex_QCQP_RHS':
        mat_name = f"random_convex_qcqp_dataset_var{args.num_var}_ineq{args.num_ineq}_eq{args.num_eq}_ex{args.data_size}"
        file_path = os.path.join('datasets', 'convex_qcqp', f"{mat_name}.mat")
        problem = Convex_QCQP(prob_type=args.prob_type, learning_type='test', file_path=file_path)
    elif args.prob_type == 'QP_RHS':
        mat_name = f"dc3_{args.num_var}_{args.num_ineq}_{args.num_eq}_{args.data_size}"
        file_path = os.path.join('datasets', 'qp', f"{mat_name}")
        problem = QP(prob_type=args.prob_type, learning_type='test', file_path=file_path)
    else:
        print("Unsupported problem type")
        return

    # Use the test split.
    batch_size = problem.test_size
    num_var   = problem.num_var
    num_ineq  = problem.num_ineq
    num_eq    = problem.num_eq
    num_lb = problem.num_lb
    num_ub = problem.num_ub
    # Total KKT system size 
    total_vars = num_var + 2 * num_ineq + num_eq + num_lb + num_ub

    # Initialize variables: x, η, s, lamba, zl0, and zu0. 

        # === Primal variables: strictly feasible by solving A x = b ===
    if num_eq > 0:
        # problem.A: [batch_size, m, n], problem.b: [batch_size, m, 1]
        # x0 = pinv(A) @ b  => shape [batch_size, n, 1]
        A_batch = problem.A                           # [B, m, n]
        b_batch = problem.b                           # [B, m, 1]
        A_pinv   = torch.linalg.pinv(A_batch)        # [B, n, m]
        x0       = torch.bmm(A_pinv, b_batch)        # [B, n, 1]
    else:
        # no equalities ⇒ start at zero
        x0 = torch.zeros((batch_size, num_var, 1), device=device)

    # === Inequality slacks & duals: enforce g(x0)+s0 = 0 strictly ===
    if num_ineq > 0:
        g0  = problem.ineq_resid(x0)                 # [B, m, 1]
        s0  = torch.clamp(-g0, min=1e-6)             # ensures g0 + s0 = 0 and s0>0
        eta0 = torch.ones_like(s0)                   # positive duals
    else:
        eta0 = torch.zeros((batch_size, 0, 1), device=device)
        s0   = torch.zeros((batch_size, 0, 1), device=device)

    # === Box constraints (unchanged if you have none) ===
    if num_lb > 0:
        x0 = torch.max(x0, problem.lb + 1e-3)
        dl0 = x0 - problem.lb
        zl0 = torch.ones_like(dl0)
    else:
        dl0 = torch.zeros((batch_size, 0, 1), device=device)
        zl0 = torch.zeros((batch_size, 0, 1), device=device)

    if num_ub > 0:
        x0 = torch.min(x0, problem.ub - 1e-3)
        du0 = problem.ub - x0
        zu0 = torch.ones_like(du0)
    else:
        du0 = torch.zeros((batch_size, 0, 1), device=device)
        zu0 = torch.zeros((batch_size, 0, 1), device=device)

    # === Equality multipliers if any (warm‑start at zero) ===
    lamb0 = torch.zeros((batch_size, num_eq, 1), device=device)

    # === Pack into big vector y ===
    y = torch.cat([x0, eta0, s0, lamb0, zl0, zu0], dim=1)

    B       = problem.test_size
    n       = problem.num_var
    m_ineq  = problem.num_ineq
    m_eq    = problem.num_eq
    m_lb    = problem.num_lb
    m_ub    = problem.num_ub
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
    

    print("Starting Damped Newton's Method with Regularization for problem type:", args.prob_type)
    print("Batch size:", batch_size)
    print("Total KKT system size (variables):", total_vars)
    print("Tolerance:", args.tol)
    print("Max iterations:", args.max_iters)
    print("Initial α:", args.init_alpha, "Regularization λ:", args.reg)
    print("----------------------------------------------------")

    lambda_hessian = 1
    mu_k   = (eta0 * s0).mean().item()    # μ₀ = average η₀·s₀
    eps_mu = mu_k             # inner‐solve tolerance
    sigma  = args.sigma                   # your centering parameter

    lambda_reg   = args.reg           # start value
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
    J, F, J1 = problem.cal_kkt_newton(x, eta, s, lamb, zl, zu, mu=0.0, sigma=args.sigma, lambs = lambda_hessian)
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
    iter_armigo = 0
    while mu_k > 1e-8 and res_norm > args.tol:
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
        
        J, F, J1 = problem.cal_kkt_newton(x, eta, s, lamb, zl, zu, mu_k, args.sigma, lambda_hessian)
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
            J_aff, F_aff, J1 = problem.cal_kkt_newton(x, eta, s, lamb, zl, zu, mu_k, 0, lambda_hessian)
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

            J, F, _ = problem.cal_kkt_newton(x, eta, s, lamb, zl, zu, mu_k, sigma_k, lambda_hessian)

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
            obj_val = problem.obj_fn(x).mean().item()
            ineq_violation = problem.ineq_dist(x).max().item() if num_ineq > 0 else 0.0
            eq_violation = problem.eq_dist(x).max().item() if num_eq > 0 else 0.0
            obj_grad_norm = problem.obj_grad(x).mean().item()
            ineq_grad_norm = problem.ineq_grad(x).mean().item()
            AT_lamb = problem.AT_lamb(x, eta, s, lamb, zl, zu, args.sigma).norm().item()
            print(f"Iter: {iter_idx:3d} | Obj: {obj_val:8.4f} | Obj Grad Norm: {obj_grad_norm:12.4f} | Res Norm: {res_norm:12.4f} |"
                f"Max_Ineq: {ineq_violation:8.4f} | Max_Eq: {eq_violation:8.4f} | Inequality Gradient: {ineq_grad_norm:12.4f} | A_transpose_lambda: {AT_lamb:12.4f}| Lambda: {lamb.mean().item():12.4f}")
            
            with torch.no_grad():
                F_blocks = torch.split(F, [num_var, num_ineq, num_ineq, num_eq, num_lb, num_ub], dim=1)
                names = ['Stationarity', 'Ineq primal', 'Ineq complementarity', 'Eq constraints', 'Lower bound comp', 'Upper bound comp']
                for name, block in zip(names, F_blocks):
                    print(f"{name} Norm: {block.norm().item():.4e}")


            if res_norm < args.tol:
                print("Convergence achieved at iteration", iter_idx)
                break

            # Use a line search to choose a damping factor α.
            alpha = args.init_alpha
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
            phi_val, grad_x, grad_s = problem.merit_phi_and_grads(x, s, mu=mu_k, nu=nu)


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
            while alpha_s > 1e-12:
                iter_armigo += 1
                x_trial = x + alpha_s * dx
                s_trial = s + alpha_s * ds

                phi_trial, _, _ = problem.merit_phi_and_grads(x_trial, s_trial,mu=mu_k, nu=nu)   # merit uses (x,s) only
                
                if phi_trial <= phi_val + args.ls_c * alpha_s * dir_deriv_scalar:
                    success = True
                    lambda_hessian = lambda_hessian / 2
                    break
                else:
                    lambda_hessian = lambda_hessian * 2
                alpha_s *= args.ls_tau                 # shrink

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
            J, F, J1 = problem.cal_kkt_newton(x, eta, s, lamb, zl, zu, mu_k, sigma_k, lambda_hessian)
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

    end_time = time.time()
    total_time = end_time - start_time

    # Final evaluation.
    x_final = y[:, :num_var, :]
    final_obj = problem.obj_fn(x_final).mean().item()
    final_res_norm = F.norm().item()  # Residual norm from last iteration.
    final_ineq = problem.ineq_dist(x_final).max().item() if num_ineq > 0 else 0.0
    final_eq = problem.eq_dist(x_final).max().item() if num_eq > 0 else 0.0

    # 1) Stationarity residual
    F1 = F[:, 0:num_var, :]                       
    dualInf = F1.abs().max().item()

    # 2) Primal infeasibility (inequalities + equalities)
    ineq_res = problem.ineq_resid(x)              # [B, m_ineq, 1]
    eq_res   = problem.eq_resid(x)                # [B, m_eq, 1]
    primInf  = torch.cat([ineq_res.abs(), eq_res.abs()], dim=1).max().item()

    # 3) Complementarity measure
    comp     = (eta * s).abs()                    # [B, m_ineq, 1]
    ispDualInf = comp.max().item() / max(primInf, 1e-12)

    print("----------------------------------------------------")
    print("Newton's Method completed")
    print("Total iterations:", total_iters)
    print("Final Objective:", final_obj)
    print("Final Residual Norm:", final_res_norm)
    print("Final Max Inequality Violation:", final_ineq)
    print("Final Max Equality Violation:", final_eq)
    print("Total Time Taken (s):", total_time)
    print("Primal feas: ", primInf)
    print("Dual feas: ", dualInf)
    print("Armigo Iterations: ", iter_armigo)
    function_evals = total_iters*2 + iter_armigo
    gradient_evals = total_iters*2
    hessian_evals = total_iters
    print("Function Evals: ", function_evals)
    print("Gradient Evals: ", gradient_evals)
    print("Hessian Evals: ", hessian_evals)

if __name__ == "__main__":
    main()
