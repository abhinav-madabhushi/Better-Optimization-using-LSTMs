#!/usr/bin/env python3
import os
import time
import torch
import configargparse
from problems.Convex_QCQP import Convex_QCQP
from problems.QP         import QP

def compute_step_direction(problem, x, eta, s, lamb, zl, zu, mu, sigma):
    """
    Build and solve the KKT system J * d = -F exactly.
    Returns the stacked direction d and the residual norm.
    """
    J, F, J1 = problem.cal_kkt(x, eta, s, lamb, zl, zu, mu, sigma)
    res_norm = F.norm().item()
    # Solve J d = -F
    d = torch.linalg.solve(J, -F)
    return d, res_norm

def main():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--prob_type', type=str, required=True,
                        help='Convex_QCQP_RHS or QP_RHS')
    parser.add_argument('--device',    type=str, default='cpu')
    # IPM parameters
    parser.add_argument('--muP0',      type=float, default=1e-4)
    parser.add_argument('--muB0',      type=float, default=1e-4)
    parser.add_argument('--sigma',     type=float, default=0.8,
                        help='fraction-to-boundary for projection (0<σ<1)')
    parser.add_argument('--armijoTol', type=float, default=1e-3)
    parser.add_argument('--gammaC',    type=float, default=0.5)
    parser.add_argument('--max_iters', type=int,   default=50)
    # Tolerances for O-iteration
    parser.add_argument('--tol_kkt',   type=float, default=1e-1,
                        help='KKT residual tolerance for O-iteration')
    parser.add_argument('--tol_prim',  type=float, default=1e-1,
                        help='Primal infeasibility tolerance for O-iteration')
    parser.add_argument('--tol_dual',  type=float, default=1e-1,
                        help='Dual infeasibility tolerance for O-iteration')
    parser.add_argument('--muL0',  type=float, default=1e-4, help='initial μL ≥ μP')
    parser.add_argument('--chi0',  type=float, default=1.0, help='initial χ^max')
    parser.add_argument('--tau0',  type=float, default=1e-2, help='initial τ for M‐iters')
    parser.add_argument('--smax',  type=float, default=1e6,   help='max for slack‐shift clamp')
    parser.add_argument('--ymax',  type=float, default=1e6,   help='max for y‐shift clamp')
    parser.add_argument('--wmax',  type=float, default=1e6,   help='max for w‐shift clamp')

    parser.add_argument('--outer_tol', type=float, default=1e-2,
                    help='termination tol on ∥∇M∥ for outer loop')

    # Data config
    parser.add_argument('--data_size', type=int, default=1000, help='Number of problem instances (data size)')
    parser.add_argument('--num_var', type=int, required=True, help='Number of decision variables')
    parser.add_argument('--num_eq', type=int, required=True, help='Number of equality constraints')
    parser.add_argument('--num_ineq', type=int, required=True, help='Number of inequality constraints')
    args = parser.parse_args()

    device = args.device
    #torch.set_default_dtype(torch.float64)

    # -- 1) Load the problem --
    if args.prob_type == 'Convex_QCQP_RHS':
        # adjust these if you need num_var/num_ineq/num_eq from command line
        file_path = os.path.join('datasets', 'convex_qcqp',
                                 f"random_convex_qcqp_dataset_var{args.num_var}"
                                 f"_ineq{args.num_ineq}_eq{args.num_eq}"
                                 f"_ex{args.data_size}.mat")
        problem = Convex_QCQP('Convex_QCQP_RHS', 'test', file_path=file_path,
                              device=device)
    elif args.prob_type == 'QP_RHS':
        file_path = os.path.join('datasets', 'qp',
                                 f"dc3_{args.num_var}_{args.num_ineq}_"
                                 f"{args.num_eq}_{args.data_size}")
        problem = QP('QP_RHS', 'test', file_path=file_path, device=device)
    else:
        raise ValueError("Unsupported problem type")

    # dims
    B = problem.test_size
    n = problem.num_var
    m = problem.num_ineq
    p = problem.num_eq
    nl = problem.num_lb
    nu = problem.num_ub

    # -- 2) Strictly-feasible warm start (as in your Newtons_method.py) --
    # x0
    if p > 0:
        A, b = problem.A, problem.b
        x0 = torch.bmm(torch.linalg.pinv(A), b)
    else:
        x0 = torch.zeros((B,n,1), device=device)
    # ineq slacks & duals
    if m > 0:
        g0   = problem.ineq_resid(x0)
        s0   = torch.clamp(-g0, min=1e-6)
        eta0 = torch.ones_like(s0)
    else:
        s0   = torch.zeros((B,0,1), device=device)
        eta0 = torch.zeros((B,0,1), device=device)
    # bounds
    if nl>0:
        x0 = torch.max(x0, problem.lb + 1e-3)
        dl0 = x0 - problem.lb
        zl0 = torch.ones_like(dl0)  
    else:
        dl0 = torch.zeros((B,0,1), device=device)
        zl0 = torch.zeros((B,0,1), device=device)
    if nu>0:
        x0 = torch.min(x0, problem.ub - 1e-3)
        du0 = problem.ub - x0
        zu0 = torch.ones_like(du0)
    else:
        du0 = torch.zeros((B,0,1), device=device)
        zu0 = torch.zeros((B,0,1), device=device)
    # equality multipliers
    lamb0 = torch.zeros((B,p,1), device=device)

    # stack into one big vector v = [x; η; s; λ; zl; zu]
    v_k = torch.cat([x0, eta0, s0, lamb0, zl0, zu0], dim=1)

    # build bL, bU for projection  (including η-block)
    bL = torch.cat([
        -torch.inf * torch.ones_like(x0),   # x unbounded
        torch.zeros_like(eta0),             # η ≥ 0
        torch.zeros_like(s0),               # s ≥ 0
        -torch.inf * torch.ones_like(lamb0),# λ unbounded
        torch.zeros_like(zl0),              # zₗ ≥ 0
        torch.zeros_like(zu0)               # zᵤ ≥ 0
    ], dim=1)
    bU = torch.cat([
        +torch.inf * torch.ones_like(x0),   # x unbounded
        +torch.inf * torch.ones_like(eta0), # η unbounded
        +torch.inf * torch.ones_like(s0),   # s unbounded
        +torch.inf * torch.ones_like(lamb0),# λ unbounded
        +torch.inf * torch.ones_like(zl0),  # zₗ unbounded
        +torch.inf * torch.ones_like(zu0)   # zᵤ unbounded
    ], dim=1)

    # 3) initialize shifts
    muP, muB = args.muP0, args.muB0
    muL      = args.muL0
    chi_max  = args.chi0
    tau      = args.tau0

    yE = lamb0.clone()
    sE = s0.clone()
    wE = eta0.clone() 

    sizes = [n, m, m, p, nl, nu]  # for torch.split

    outer = 0
    while True:
        # --- 0) Check ∥∇M(v_k)∥ and break if small ---
        # unpack current v_k
        x, eta, s, lamb, zl, zu = torch.split(v_k, sizes, dim=1)
        # enable grads
        for T in (x, eta, s, lamb):
            T.requires_grad_(True)
        M0 = problem.merit(x,eta,s,lamb, yE,sE,wE, muP, muB).mean()
        # 1) compute the full gradient ∇M/∂v
        gradM = problem.merit_grad(
            x, eta, s, lamb,   # primal & dual blocks
            yE, sE, wE,        # shifts
            muP, muB           # parameters
        ).squeeze(-1)          # shape [B, total_vars]

        # 2) split into the four blocks we care about for stopping:
        grad_x, grad_eta, grad_s, grad_lamb, *_ = torch.split(gradM, sizes, dim=1)

        # 3) compute the infinity‐norm over all those blocks:
        norm_x   = grad_x  .abs().max().item()
        norm_eta = grad_eta.abs().max().item()
        norm_s   = grad_s  .abs().max().item()
        norm_l   = grad_lamb.abs().max().item()
        grad_norm = max(norm_x, norm_eta, norm_s, norm_l)

        # 4) use grad_norm for your outer‐loop stopping test
        if grad_norm < args.outer_tol:
            print(f"Converged outer: ∥∇M∥={grad_norm:.2e}")
            break
        total_iters = 0
        for it in range(args.max_iters):
            # unpack
            x, eta, s, lamb, zl, zu = torch.split(v_k, sizes, dim=1)

            # 1) step direction
            d, res_norm = compute_step_direction(
                problem, x, eta, s, lamb, zl, zu, muP, args.sigma)
            
            # Diagnostics.
            obj_val = problem.obj_fn(x).mean().item()
            ineq_violation = problem.ineq_dist(x).max().item() if m > 0 else 0.0
            eq_violation = problem.eq_dist(x).max().item() if p > 0 else 0.0
            obj_grad_norm = problem.obj_grad(x).mean().item()
            ineq_grad_norm = problem.ineq_grad(x).mean().item()
            print(f"Iter: {it:3d} | Obj: {obj_val:8.4f} | Obj Grad Norm: {obj_grad_norm:12.4f} | Res Norm: {res_norm:12.4f} |"
                f"Max_Ineq: {ineq_violation:8.4f} | Max_Eq: {eq_violation:8.4f} | Inequality Gradient: {ineq_grad_norm:12.4f} ")

            # 2) evaluate merit & KKT‐norm at current v_k
            M0_tensor = problem.merit(x, eta, s, lamb,
                               yE, sE, wE,
                               muP, muB)
            M0_val    = M0_tensor.mean().item()

            J0, F0, J1 = problem.cal_kkt(
                x, eta, s, lamb, zl, zu, muP, args.sigma)
            kkt0 = F0.norm().item()

            # 3) projected-search backtracking
            alpha = 1.0
            total_iters += 1
            while True:
                v_trial = v_k + alpha*d
                # project to Ω_k
                lower = bL + args.sigma*muB
                upper = bU - args.sigma*muB
                v_proj = torch.max(torch.min(v_trial, upper), lower)

                # unpack projected
                xp, etap, sp, lambp, zlp, zup = torch.split(v_proj, sizes, dim=1)

                # merit & residual at trial
                Mp_tensor = problem.merit(xp, etap, sp, lambp,
                                   yE, sE, wE,
                                   muP, muB)
                Mp_val    = Mp_tensor.mean().item()
                Jp, Fp, J1 = problem.cal_kkt(
                    xp, etap, sp, lambp, zlp, zup,
                    muP, args.sigma)
                kktp = Fp.norm().item()

                # sufficient KKT‐reduction?
                condF = (kktp <= kkt0)
                # directional derivative of M in direction d
                gradM = problem.merit_grad(x, eta, s, lamb, yE, sE, wE, muP, muB)
                dirDer = (gradM * d).sum().item()
                # Armijo on M?  (use the scalar means, not the full tensors)
                condM = (Mp_val <= M0_val + args.armijoTol * alpha * dirDer)
                #delta_norm = (v_proj - v_k).norm().item()
                #print(f" ls #{it}: α={alpha:.2e},  kktp={kktp:.2e},  condF={condF},  condM={condM},  ∥Δv∥={delta_norm:.2e}")
                if condF or condM:
                    print("Accept step")
                    v_k = v_proj

                    M_L0 = problem.merit(x,eta,s,lamb,yE,sE,wE, muL, muB).mean()
                    # reuse v_trial, dirDer, alpha from line‐search
                    xp, etp, sp, lmbp, _zlp, _zup = torch.split(v_trial, sizes, dim=1)
                    M_Lp = problem.merit(xp,etp,sp,lmbp, yE,sE,wE, muL, muB).mean()
                    if M_Lp <= M_L0 + args.armijoTol * alpha * dirDer:
                        muL = max(muL, muP)
                    else:
                        muL = max(0.5*muL, muP)
                    break
                alpha *= args.gammaC

            # unpack the newly accepted point
            x, eta, s, lamb, zl, zu = torch.split(v_k, sizes, dim=1)
            J, F, J1   = problem.cal_kkt(
                x, eta, s, lamb, zl, zu, muP, args.sigma)
            kkt_norm = F.norm().item()

            # primal infeasibility
            ineq_violation = problem.ineq_dist(x).clamp(min=0).max().item()
            eq_violation   = problem.eq_dist(x).abs().max().item()
            prim_inf = max(ineq_violation, eq_violation)

            # dual infeasibility ≈ gradient residual for x‐KKT
            F_mat = F.squeeze(-1)
            grad_res = F_mat[:, :n]
            dual_inf = grad_res.abs().max().item()

            print(f"[O{outer:1d} I{it:2d}] obj={problem.obj_fn(x).mean().item():9.4f} "
                  f"| kkt={kkt_norm:9.2e} | prim_inf={prim_inf:9.2e} | dual_inf={dual_inf:9.2e}")

            if (kkt_norm <= args.tol_kkt
                and prim_inf  <= args.tol_prim
                and dual_inf  <= args.tol_dual):
                # -- O‐iterate updates --
                yE = lamb.detach().clone()
                sE = s   .detach().clone()
                wE = eta .detach().clone()
                chi_max *= 0.5
                tau     *= 0.5
                iter_type = 'O'

            else:
                for T in (x, eta, s, lamb):
                    T.requires_grad_(True)
                M0 = problem.merit(x,eta,s,lamb,yE,sE,wE,muP,muB).mean()
                gradM = problem.merit_grad(x, eta, s, lamb, yE, sE, wE, muP, muB)
                # drop the singleton last dim
                gradM = gradM.squeeze(-1)  
                # split into the six blocks [x, η, s, λ, zₗ, zᵤ]
                grad_x, grad_eta, grad_s, grad_lamb, grad_zl, grad_zu = torch.split(gradM, sizes, dim=1)
                # now get ∞-norms
                norm_x   = grad_x  .abs().max().item()
                norm_s   = grad_s  .abs().max().item()
                norm_y   = grad_lamb.abs().max().item()    # “y” block is lamb
                norm_w   = grad_eta.abs().max().item()     # “w” block is eta
                                # M‐iteration?
                if (norm_x <= tau
                    and norm_s <= tau
                    and norm_y <= tau*muP
                    and norm_w <= tau*((s+muB)/(eta+muB)).max()):
                    # -- M‐iterate updates --
                    chi_max, tau = chi_max, 0.5*tau
                    sE = torch.clamp(s,    min=0,        max=args.smax)
                    yE = torch.clamp(lamb, min=-args.ymax, max=args.ymax)
                    wE = torch.clamp(eta,  min=0,        max=args.wmax)
                    # update muP
                    muP = muP if prim_inf <= tau else 0.5*muP
                    # update muB
                    if torch.min(s+muB, eta+muB).item() >= -tau:
                        muB = muB
                    else:
                        muB = 0.5*muB
                    iter_type = 'M'
                else:
                    # F‐iteration: no updates
                    iter_type = 'F'

            print(f"Iteration type: {iter_type}")

            # check convergence
            if kkt_norm < args.tol_kkt:
                print("Converged at outer", outer, "inner", it)
                break

        # reduce barrier / penalty parameters
        muP *= 0.1
        muB *= 0.1
        outer += 1

    # final solution
    x_sol = v_k[:, :n, :]
    print("Final x:", x_sol)
    print("Total iterations:", total_iters)

if __name__ == '__main__':
    main()
