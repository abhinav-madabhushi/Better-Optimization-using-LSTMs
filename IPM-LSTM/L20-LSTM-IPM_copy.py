import argparse, time, torch, torch.optim as optim
import torch.nn.init as init

import os
from problems.Convex_QCQP import Convex_QCQP        
from problems.QP_copy import QP
from problems.QP_copy import BoxQP
from models.LSTM_L20_Projected_Math import PS_L20_LSTM         

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

#DIM_X    = 10        # decision variables
DIM_S    = 50         # inequality slacks
HIDDEN   = 128        # LSTM hidden size, did hidden size 32 make the model better? can play around with this
K_INNER  = 1         # micro-steps per outer iteration, need to fix this in the code

EPOCHS   = 20                   # meta-training epochs
LR_META  = 1e-4                   # learning-rate for Adam

MAX_OUTER  = 500
tolP, tolD, tolC = 1e-4, 1e-4, 1e-4
TOL = 1e-4
S_MAX_E = 1e2  
Y_MAX_E = 1e2   
W_MAX_E = 1e2 

TOLM = 0.1

Z1_MAX_E = 1e2  
Z2_MAX_E = 1e2   
X1_MAX_E = 1e2
X2_MAX_E = 1e2    

# def project_tube_all(problem,
#                     xk, sk,
#                     yk, vk, z1k, z2k, wk,
#                     x, s,
#                     y, vv, z1, z2, w,
#                     x1E, x2E, sE, yE, vE, z1E, z2E, wE,
#                     muP, muB,
#                     sigma: float = 0.80,
#                     eps=1e-6):

#     B, device = x.shape[0], x.device

#     # collapse singleton dims to [B,n]
#     xk  = xk .squeeze(-1)
#     z1k = z1k.squeeze(-1)
#     z2k = z2k.squeeze(-1)

#     x  = x .squeeze(-1)
#     z1 = z1.squeeze(-1)
#     z2 = z2.squeeze(-1)

#     n = xk.shape[1]

#     # pull out lb, ub as [B,n]
#     lb = problem.lb
#     if lb.dim()==1:
#         lb = lb.unsqueeze(0).expand(B, n)
#     ub = problem.ub
#     if ub.dim()==1:
#         ub = ub.unsqueeze(0).expand(B, n)

#     # masks for which bounds exist
#     mask_lb = torch.isfinite(lb)   # [B,n]
#     mask_ub = torch.isfinite(ub)   # [B,n]

#     # --- project x ---
#     neginf = float('-inf')
#     posinf = float('inf')

#     lower_x = torch.where(mask_lb,
#                           lb - muB + eps,
#                           neginf)        # [B,n]
#     upper_x = torch.where(mask_ub,
#                           ub + muB - eps,
#                           posinf)        # [B,n]

#     x_p = torch.max(torch.min(x, upper_x), lower_x) \
#               .unsqueeze(-1)               # [B,n,1]

#     # --- project z1 ---
#     floor_z1 = torch.where(mask_lb,
#                            -muB + eps,
#                            neginf)       # [B,n]
#     z1_p = torch.max(z1, floor_z1) \
#                .unsqueeze(-1)            # [B,n,1]

#     # --- project z2 ---
#     floor_z2 = torch.where(mask_ub,
#                            -muB + eps,
#                            neginf)       # [B,n]
#     z2_p = torch.max(z2, floor_z2) \
#                .unsqueeze(-1)            # [B,n,1]

#     # pass others through
#     return x_p, s, y, vv, z1_p, z2_p, w

def project_tube_all(
        problem,
        xk, sk, yk, vk, z1k, z2k, wk,           # current iterate v_k
        x,  s,  y,  vv, z1,  z2,  w,            # trial point  v
        x1E, x2E, sE, yE, vE, z1E, z2E, wE,     # (unused here)
        muP, muB,
        sigma: float = 0.80, 
        eps = 1e-6): # look at pdProj code and what it does to prevent numerical instability
    """
    Projection onto the perturbed tube   Ω_k  of the paper (only box parts).

    Guarantees, for every component j that has a finite bound,

        x_pj  - lb_j + μB >= 0      (lower tube wall)
        ub_j  - x_pj + μB >= 0      (upper tube wall)
        z1_pj + μB      > 0
        z2_pj + μB      > 0
    """

    # --- generic lower-wall rule, valid only when ell ≤ 0 --------------
    def proj_lower(v, v_k, ell):
        # lower_bound = v_k - ell            # [B, …] same shape as v
        # return v.clamp(min=lower_bound)
        #wall = -muB.unsqueeze(-1).expand_as(v_k)
        wall = torch.minimum(v_k - sigma * (v_k - ell) , torch.full_like(v_k, 0))
        lower = wall 
        return torch.maximum(v, lower)

    B, device = x.shape[0], x.device
    n         = x.shape[1]

    # collapse last singleton dimension if present
    xk  = xk.squeeze(-1);   x  = x.squeeze(-1)
    z1k = z1k.squeeze(-1); z1 = z1.squeeze(-1)
    z2k = z2k.squeeze(-1); z2 = z2.squeeze(-1)


    # ------------------------------------------------------------------ #
    # physical box bounds for x
    # ------------------------------------------------------------------ #
    lb = problem.lb
    ub = problem.ub
    if lb.dim() == 1:
        lb = lb.unsqueeze(0).expand(B, n)
    if ub.dim() == 1:
        ub = ub.unsqueeze(0).expand(B, n)

    mask_lb = torch.isfinite(lb)          # [B,n] – where a lower bound exists
    mask_ub = torch.isfinite(ub)          # [B,n] – where an upper bound exists

    x1 = torch.where(mask_lb, x - lb, 0)
    x2 = torch.where(mask_ub, ub - x, 0)

    x1k = torch.where(mask_lb, xk - lb, 0)
    x2k = torch.where(mask_ub, ub - xk, 0)

    x1_proj = torch.where(mask_lb, proj_lower(x1, x1k, -muB), x1) 
    x2_proj = torch.where(mask_ub, proj_lower(x2, x2k, -muB), x2) 

    tol = 1e-6

    mask1 = (x1_proj - x1).abs() > tol # x1_proj != x1
    mask2 = (~mask1) & ((x2_proj - x2).abs() > tol) # x2_proj != x2

    p1 = x1_proj + lb
    p2 = ub - x2_proj

    x_flat = torch.where(mask1, p1,
                torch.where(mask2, p2, x))
    
    x_p = x_flat.unsqueeze(-1)

    #s_p = proj_lower_negwall(s.squeeze(-1),  sk.squeeze(-1), ell_sw).unsqueeze(-1)
    #w_p = proj_lower_negwall(w.squeeze(-1),  wk.squeeze(-1), ell_sw).unsqueeze(-1)

    z1_p_raw = torch.where(mask_lb,
                        proj_lower(z1,  z1k, -muB),
                        z1)                # shape [B,n]
    z1_p = torch.clamp(z1_p_raw, max=1e3)  # clamp above at 1e10
    z1_p = z1_p.unsqueeze(-1)               # shape [B,n,1]
    # two‐step version for clarity
    z2_p_raw = torch.where(mask_ub,
                        proj_lower(z2, z2k, -muB),
                        z2)                # shape [B,n]
    z2_p = torch.clamp(z2_p_raw, max=1e3)  # clamp above at 1e10
    z2_p = z2_p.unsqueeze(-1)               # shape [B,n,1]


    # y and vv (multipliers for general eq/ineq) have no tube walls
    return x_p, s, y, vv, z1_p, z2_p, w





# def frac_to_boundary(problem,
#                      x, s, y, w,
#                      x_k, s_k, y_k, w_k,
#                      muB,
#                      tau=0.995,
#                      eps=1e-8):
#     """
#     Fraction-to-boundary step for [x,s,y,w] with:
#       x in [lb, ub],
#       s >= –muB+eps,
#       w >= –muB+eps,
#     all via α = τ * min_i α_i, α_i coming from each bound.
#     """

#     # 1) squeeze off the last singleton → [B,d]
#     B, n, _ = x.shape
#     m        = s.shape[1]
#     x = x.view(B, n);   xk = x_k.view(B, n)
#     s = s.view(B, m);   sk = s_k.view(B, m)
#     y = y.view(B, m);   yk = y_k.view(B, m)
#     w = w.view(B, m);   wk = w_k.view(B, m)

#     # 2) compute deltas
#     dx = x - xk
#     ds = s - sk
#     dy = y - yk
#     dw = w - wk

#     device = x.device
#     inf = float("inf")

#     # 3) start α = 1 for each batch
#     alpha = torch.ones(B, device=device)

#     # 4) x box-bounds
#     lb = problem.lb
#     ub = problem.ub
#     lb = torch.as_tensor(lb, device=device, dtype=x.dtype)
#     ub = torch.as_tensor(ub, device=device, dtype=x.dtype)
#     # if scalars, expand to vectors
#     if lb.numel() == 1: lb = lb.expand(n)
#     if ub.numel() == 1: ub = ub.expand(n)
#     # shape → [B,n]
#     lb = lb.view(1, n).expand(B, n)
#     ub = ub.view(1, n).expand(B, n)

#     # candidates for x
#     c_low  = torch.where(dx <  0, (lb - xk) / dx,  torch.full_like(dx, +inf))
#     c_high = torch.where(dx >  0, (ub - xk) / dx,  torch.full_like(dx, +inf))
#     a_x = torch.min(c_low.min(dim=1).values,
#                     c_high.min(dim=1).values)
#     alpha = torch.min(alpha, a_x)

#     # 5) s lower bound: s >= -muB + eps
#     b_s = -muB + eps
#     c_s = torch.where(ds < 0,
#                       (b_s - sk) / ds,
#                       torch.full_like(ds, +inf))
#     if c_s.dim() == 3 and c_s.size(0) == 1:
#         c_s = c_s.squeeze(0)
#     a_s = c_s.min(dim=1).values
#     alpha = torch.min(alpha, a_s)

#     # 6) w lower bound: w >= -muB + eps
#     b_w = -muB + eps
#     c_w = torch.where(dw < 0,
#                       (b_w - wk) / dw,
#                       torch.full_like(dw, +inf))
#     if c_w.dim() == 3 and c_w.size(0) == 1:
#         c_w = c_w.squeeze(0)
#     a_w = c_w.min(dim=1).values
#     alpha = torch.min(alpha, a_w)

#     # 7) scale by τ and clamp into [0,1]
#     alpha = (alpha * tau).clamp(0.0, 1.0).view(B, 1)

#     # 8) take the step
#     x_new = xk + alpha * dx
#     s_new = sk + alpha * ds
#     y_new = yk + alpha * dy
#     w_new = wk + alpha * dw

#     # 9) restore the trailing 1‐dim
#     return (
#         x_new.view(B, n, 1),
#         s_new.view(B, m, 1),
#         y_new.view(B, m, 1),
#         w_new.view(B, m, 1),
#     )







def solve_one_qcqp(meta_opt, problem: "QP",
                   net: PS_L20_LSTM,
                   muP0=1e-4, muB0=1,
                   max_outer=MAX_OUTER, train_test = 'train', val_stop = 5, print_level = 0):
    
    # change muB to 1e-2: done
    total_start = time.perf_counter()
    """
    Projected-search IPM loop powered by a diagonal L20-LSTM.
    Returns final (x,s,y,w) and the average inner loss.
    """

    # loop through batches here

    if train_test == 'train':
        print("train")
    else: 
        print("test")
    
    if train_test != "train":
        muB0 = 1e-2

    B = problem.Q.shape[0]
    n_vec = problem.n_vec
    n, m_ineq, m_eq = problem.num_var, problem.num_ineq, problem.num_eq
    device = DEVICE

    val = 0

    # initializing the variables, will need initialize v also here later 
    if m_eq > 0:
        A_pinv = torch.linalg.pinv(problem.A)       # [B,n,p]
        x  = torch.bmm(A_pinv, problem.b, device=device)        # [B,n,1]
    else:
        x  = torch.zeros(B, n, 1, device=device)
        v = None
        vE = None

    # slack variable s initialization for the inequalities
    if m_ineq > 0:
        g0    = problem.ineq_resid(x)           # [B,m_ineq,1]
        s = torch.clamp(-g0, min=1e-6, device=device)           # [B,m_ineq,1] 
        muP = x.new_tensor([muP0], device=device)   
    else:
        s = None
        y = None
        w = None
        sE = None
        yE = None
        wE = None

    eps = 1e-20

    # initialize the duals anbd the shifts            
    muB = x.new_tensor([muB0], device=device)
    muP = x.new_tensor([muP0], device=device)
    #tau_k = 2
    tolM = torch.tensor(TOLM).to(device)

    # initializing x using the bounds
    if problem.num_lb != 0 and problem.num_ub != 0:
        # initialize x
        lb_vals = problem.lb.expand(B, n).to(device)    # [B,n]
        ub_vals = problem.ub.expand(B, n).to(device)    # [B,n]

        # masks for which bounds are finite
        has_lb = torch.isfinite(lb_vals).to(device)                    # [B,n]
        has_ub = torch.isfinite(ub_vals).to(device)                    # [B,n]

        # start with zeros
        x_init = torch.zeros((B, n), device=device)       # [B,n]

        # both bounds → midpoint
        both = has_lb & has_ub
        #both = both.to(device)
        x_init[both] = 0.5 * (lb_vals[both] + ub_vals[both])

        # lb only → lb + 1
        lb_only = has_lb & ~has_ub 
        x_init[lb_only] = lb_vals[lb_only] + 1.0

        # ub only → ub − 1
        ub_only = has_ub & ~has_lb
        x_init[ub_only] = ub_vals[ub_only] - 1.0
        x = x_init.unsqueeze(-1)
        xE = x.clone()

        # initialize x1, x2, z1, and z2
        lb = problem.lb.view(B, n, 1).expand(B, n, 1).to(device)   # [B,n,1]
        ub = problem.ub.view(B, n, 1).expand(B, n, 1).to(device)    # [B,n,1]

        # masks for where bounds are finite
        mask_lb = torch.isfinite(lb)   # [B,n,1], True where ℓ_j > -∞
        mask_ub = torch.isfinite(ub)   # [B,n,1], True where u_j < +∞

        # floor tensor
        eps_t   = torch.tensor(eps, device=device)

        # --- lower‐bound slack & dual ---
        # slack  x1_j = clamp(x_j - ℓ_j, min=eps)
        raw_x1 = (x - lb).clamp(min=eps_t)               # [B,n,1]
        x1     = torch.where(mask_lb, raw_x1,
                            torch.zeros_like(raw_x1))
        x1 = x1.to(device)
        # dual   z1_j = μB / x1_j
        z1     = torch.where(mask_lb,
                            muB / x1,
                            torch.zeros_like(x1))
        
        z1 = z1.to(device)
        # --- upper‐bound slack & dual ---
        # slack  x2_j = clamp(u_j - x_j, min=eps)
        raw_x2 = (ub - x).clamp(min=eps_t)               # [B,n,1]
        x2     = torch.where(mask_ub, raw_x2,
                            torch.zeros_like(raw_x2))
        x2 = x2.to(device)
        # dual   z2_j = μB / x2_j
        z2     = torch.where(mask_ub,
                            muB / x2,
                            torch.zeros_like(x2))
        z2 = z2.to(device)
        
        x1E, x2E, z1E, z2E = x1.clone(), x2.clone(), z1.clone(), z2.clone()
        x1E.to(device), x2E.to(device), z1E.to(device), z2E.to(device)
        x1_max_e = torch.tensor(X1_MAX_E, device=device)
        x2_max_e = torch.tensor(X2_MAX_E, device=device)
        z1_max_e = torch.tensor(Z1_MAX_E, device=device)
        z2_max_e = torch.tensor(Z2_MAX_E, device=device)

    

    # # building c(x) = [g(x); A x - b] : [B,m_tot,1]
    # if problem.num_ineq>0:
    #     g_col = problem.ineq_resid(x_col)                  # [B,m_ineq,1]
    # else:
    #     g_col = torch.zeros((B,0,1), device=device)
    # if problem.num_eq>0:
    #     eq_col = torch.bmm(problem.A, x_col) - problem.b      # [B,m_eq,1]
    # else:
    #     eq_col = torch.zeros((B,0,1), device=device)
    # c_val = torch.cat([g_col, eq_col], dim=1)           # [B,m_tot,1]

    # resid_c = (c_val + s_full)     # [B,m_tot,1]
    # y_full   = resid_c / muP                         # [B,m_tot,1]
    # w_full   = muB / (s_full + muB) - y_full           # [B,m_tot,1]

    # sE, yE, wE = s_full.clone(), y_full.clone(), w_full.clone()
    # s_max_e = torch.tensor(S_MAX_E, device=device)
    # y_max_e = torch.tensor(Y_MAX_E, device=device)
    # w_max_e = torch.tensor(W_MAX_E, device=device)

    muB = muB.expand(B, 1)
    tolM = tolM.expand(B)


    L = net.lstm_px.num_layers
    H = net.lstm_px.hidden_size

    BN = B * n
    h = torch.zeros(L, BN, H, device=device)
    c = torch.zeros(L, BN, H, device=device)
    state = (h, c)

    # Xavier‐uniform initialization for the hidden and cell state of LSTM
    init.xavier_uniform_(h)
    init.xavier_uniform_(c)
    lstm_states = (h, c)

    N = 0
    parts = []

    # helper to squeeze and register a block
    def _add_block(var, name):
        nonlocal N
        if var is not None:
            flat = var.squeeze(-1)       # [B, dim]
            parts.append(flat)
            N = N + flat.shape[1]           # accumulate width

    # always include x
    _add_block(x,  'x')

    # slack/dual for bounds
    # _add_block(x1, 'x1')   # lower‐slack
    # _add_block(x2, 'x2')   # upper‐slack
    _add_block(z1, 'z1')   # lower‐dual
    _add_block(z2, 'z2')   # upper‐dual

    # inequality slacks/duals
    _add_block(s,  's')    # ineq‐slack
    _add_block(w,  'w')    # ineq‐dual
    _add_block(y,  'y')    # ineq‐multiplier

    # equality multiplier
    _add_block(v,  'v')    # eq‐multiplier

    # creating v0 according to the math
    v0 = torch.cat(parts, dim=1)    
    v0 = v0.to(device)
    
    # r_k includes μP, μB
    r_k = v0.clone() 
    r_k = r_k.to(device)     

    slack1 = x1.squeeze(-1) + muB        # should stay > 0
    slack2 = x2.squeeze(-1) + muB        # should stay > 0

    # 2) build a mask of infeasible entries
    bad_x1 = (slack1 <= 0)   # shape [B,n,1], True wherever x1+μB ≤ 0
    bad_x2 = (slack2 <= 0)   # shape [B,n,1], True wherever x2+μB ≤ 0

    total_infeas = int(bad_x1.sum().item())
    #print(f"Total infeasible variables wrt to x1 in batch: {total_infeas}")

    slackz1 = (z1.squeeze(-1) + muB)    # [B,n]
    slackz2 = (z2.squeeze(-1) + muB)    # [B,n]

    # build masks of the exact indices that went non-positive
    bad_z1 = (slackz1 <= 0).unsqueeze(-1)  # [B,n,1]
    bad_z2 = (slackz2 <= 0).unsqueeze(-1)  # [B,n,1]

    # Option A: out-of-place with torch.where
    z1 = torch.where(bad_z1, torch.zeros_like(z1), z1)
    z2 = torch.where(bad_z2, torch.zeros_like(z2), z2)             

    total_inner_loss = 0.0

    # display merit/residual
    M_k = problem.merit_M(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB, bad_x1, bad_x2, bad_z1, bad_z2).mean()
    
    M_max = problem.merit_M(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB, bad_x1, bad_x2, bad_z1, bad_z2)
    
    #total_inner_loss = total_inner_loss + M_k

    # print("x: ", x0_flat.mean().item())
    # print("s: ", s0_flat.mean().item())
    # print("y: ", y0_flat.mean().item())
    # print("w: ", w0_flat.mean().item())
    

    obj_fcn = problem.obj_fn(x).mean().item()

    #print("Objective function value: ", obj_fcn)
        
    # # diagnostics
    # s_plus = (s + muB).view(-1)                         # [B·m]
    # w_plus = (w + muB).view(-1)                         # [B·m]
    # d_val  = (c_val + s_full).view(-1)   # c(x)–s,  [B·m]

    # print(f"[dbg] s+μB min {s_plus.min():.3e} max {s_plus.max():.3e} | "
    #     f"w+μB min {w_plus.min():.3e} max {w_plus.max():.3e} | "
    #     f"c-s min {d_val.min():.3e} max {d_val.max():.3e}")

    chi = problem.chi(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB)
    
    chi_max = torch.full(chi.shape, 1000.0, device=chi.device, dtype=chi.dtype)
    
    
    if (device == 'cpu') & (print_level == 3):
        print(f"[IPM‐iter {0}] μP={muP.item():.1e}, "
                f"μB={muB.mean().item():.1e}, M={M_k:.4e}, ∥r∥={chi_max.mean().item():.4e}")
        
        for i in range(min(5, lb.shape[0])):
            print(f"obj[{i}] = {problem.obj_fn(x).view(B)[i]:.6f}")

        for i in range(min(5, lb.shape[0])):
            print(f"Problem {i}:")
            for j in range(lb.shape[1]):
                print(f"  Variable {j}: lower = {lb[i, j]}, upper = {ub[i, j]}")
            print()

    # NEED TO ADD DIAGNOSTICS FOR X1, X2, Z1, Z2
    
    # grad_x, grad_s_full, grad_y_full, grad_w_full = problem.merit_grad_M(
    #         x_col, s_full, y_full, w_full, sE, yE, wE, muP, muB
    #     )

    P = problem.primal_feasibility(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB)      
    # dual feasibility
    D = problem.dual_feasibility(x, x1, x2, s,      
            y, v, z1, z2, w,    
            x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
            muP, muB)        
    # complementarity
    C = problem.complementarity(x, x1, x2, s,      
            y, v, z1, z2, w,    
            x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
            muP, muB)    
    if (device == 'cpu') & (print_level == 3):
        print("Primal Feasability: ", P.mean())
        print("Dual Feasability: ", D.mean()) # stationarity is giving me issues
        print("Complimentarity: ", C.mean())
    
    # # flatten gradients
    # grad_x = grad_x.view(B,   n)
    # grad_s = grad_s_full.view(B, m_tot)
    # grad_y = grad_y_full.view(B, m_tot)
    # grad_w = grad_w_full.view(B, m_tot)

    # inf_x = grad_x.abs().amax(dim=1)   # [B]
    # inf_s = grad_s.abs().amax(dim=1)   # [B]
    # inf_y = grad_y.abs().amax(dim=1)   # [B]
    # inf_w = grad_w.abs().amax(dim=1)   # [B]

    # print(inf_x.mean().item())
    # print(inf_s.mean().item())
    # print(inf_y.mean().item())
    # print(inf_w.mean().item())

    ds_start = time.perf_counter()
    

    # outer IPM iterations
    for outer in range(1, max_outer+1):

        
        # flatten for LSTM call
        x = x.squeeze(-1)   
        x1 = x1.squeeze(-1)   
        x2= x2.squeeze(-1)
        z1 = z1.squeeze(-1)
        z2 = z2.squeeze(-1)

        # x  = x.requires_grad_(True)
        # x1  = x1.requires_grad_(True)
        # x2  = x2.requires_grad_(True)
        # z1 = z1.requires_grad_(True)
        # z2 = z2.requires_grad_(True)

        m = 3 * n

        if n_vec is not None:
            # per-sample true dimensions
            base = (torch.arange(n, device=device)[None, :] < n_vec[:, None]).float()  # [B,n]
            mask_x = base
            mask_z1 = base
            mask_z2 = base
            mask_v = torch.cat([mask_x, mask_z1, mask_z2], dim=1)                           # [B,m]
            mask_H = mask_v.unsqueeze(2) * mask_v.unsqueeze(1)                              # [B,m,m]
        else:
            # fixed dimension → just ones everywhere
            mask_x = torch.ones(B, n, device=device)
            mask_z1 = mask_x
            mask_z2 = mask_x
            mask_v = torch.ones(B, m, device=device)
            mask_H = torch.ones(B, m, m, device=device)

        # currently does 1 micro-iteration of the LSTM, can change that 
        (x, x1, x2, s, y, v, z1, z2, w), step_loss_vector, (lstm_states, r_k) = net(
            problem,
            x, x1, x2, s, y, v, z1, z2, w,
            x1E.squeeze(-1), x2E.squeeze(-1), sE, yE, vE, z1E.squeeze(-1), z2E.squeeze(-1), wE,
            muP, muB, r_k,
            lstm_states, bad_x1, bad_x2, bad_z1, bad_z2, M_max, n_vec = n_vec, 
            project_step=project_tube_all, outer = outer, print_level = print_level
        )

        # obtaining each of the gradients
        grad_x, grad_s, grad_y, grad_vv, grad_z1, grad_z2, grad_w = problem.merit_grad_M(
            x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB, bad_x1, bad_x2, bad_z1, bad_z2
        )
        # flatten gradients
        if grad_x is not None: 
            grad_x = grad_x.view(B,   n)
        # if grad_x1 is not None:
        #     grad_x1 = grad_x1.view(B, n)
        # if grad_x2 is not None:
        #     grad_x1 = grad_x1.view(B, n)
        if grad_z1 is not None:
            grad_z1 = grad_z1.view(B, n)
        if grad_z2 is not None:
            grad_z2 = grad_z2.view(B, n)
        
        l2_x  = ((grad_x**2)  * mask_x).sum(dim=1).sqrt()
        l2_z1 = ((grad_z1**2) * mask_z1).sum(dim=1).sqrt()
        l2_z2 = ((grad_z2**2) * mask_z2).sum(dim=1).sqrt()


        step_loss = step_loss_vector.mean()

        #step_loss = step_loss + (l2_x + l2_z1 + l2_z2).sum()

        # gx, gx1, gx2, gz1, gz2 = torch.autograd.grad(
        #     step_loss, [x, x1, x2, z1, z2],
        #     retain_graph=True,
        #     allow_unused=True  # returns None if a tensor is not in the graph
        # )
        # print("gx:", gx, "gx1:", gx1, "gx2:", gx2, "gz1:", gz1, "gz2:", gz2)

        if (device == 'cpu') & (print_level == 3):
            print("Loss for this step: ", step_loss)

        total_inner_loss = total_inner_loss + step_loss
        total_inner_loss = total_inner_loss/n

        P = problem.dual_feasibility(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB) 

        D = problem.dual_feasibility(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB) 
        C = problem.complementarity(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB) 
        
        total_inner_loss = total_inner_loss + ((P.sum() + D.sum() + C.sum())/n)

        # higher dimensions not working, why?

        #total_inner_loss = D.sum() + C.sum()
             
        
        # if train_test == 'train':
        #     meta_opt.zero_grad(set_to_none=True)      # clear old gradients
        #     step_loss.backward()                      # back-prop through the micro step
        #     torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        #     meta_opt.step()                           # update the LSTM weights
        # else:
        #     val_flag = False

        x = x.detach()
        x1 = x1.detach()
        x2 = x2.detach()
        z1 = z1.detach()
        z2 = z2.detach()
        # y = y.detach()
        # w = w.detach()
        # v = v.detach()
        # s = s.detach()

        x1E = x1E.detach(); x2E = x2E.detach();z1E = z1E.detach();z2E = z2E.detach()
        #sE = sE.detach(); yE = yE.detach(); wE = wE.detach(); vE = vE.detach()
        muB = muB.detach(); muP = muP.detach()

        


        # meta_opt.zero_grad(set_to_none=True)      # clear old gradients
        # step_loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        # #torch.autograd.set_detect_anomaly(True)
        # meta_opt.step()                           # update the LSTM weights

        if train_test == 'train':
        #     if outer % 10 == 0:
            meta_opt.zero_grad(set_to_none=True)      # clear old gradients
            #total_inner_loss = total_inner_loss/20
            total_inner_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            #torch.autograd.set_detect_anomaly(True)
            meta_opt.step()                           # update the LSTM weights

            total_inner_loss = 0

            h, c = lstm_states
            lstm_states = (h.detach(), c.detach())
            r_k = r_k.detach()
        

        # reshape back to columns
        x = x.unsqueeze(-1)   
        x1 = x1.unsqueeze(-1)   
        x2= x2.unsqueeze(-1)
        z1 = z1.unsqueeze(-1)
        z2 = z2.unsqueeze(-1)

        #print("z1: ", z1[0][0][0])

        # display merit/residual
        M_k_new = problem.merit_M(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB, bad_x1, bad_x2, bad_z1, bad_z2).mean().item()
        # M_k_should_be = problem.merit_M(x, x1, x2, s,      
        #         y, v, torch.tensor([[[z1[0][0][0]], [2.3896]]], device=device), z2, w,    
        #         x1E, x2E, sE, yE, vE, torch.tensor([[[z1[0][0][0]], [2.3896]]], device=device), z2E, wE, 
        #         muP, muB, bad_x1, bad_x2, bad_z1, bad_z2).mean()
        
        f_val, lb_1, lb_2, lb_3, lb_4, AL_1, ub_1, ub_2, ub_3, ub_4, AL_2 = problem.merit_M_indi(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB, bad_x1, bad_x2, bad_z1, bad_z2)
        if (device == 'cpu') & (print_level == 3):
            print("f_val: ", f_val.mean().item())
            print("lb_1: ", lb_1.mean().item())
            print("lb_2: ", lb_2.mean().item())
            print("lb_3: ", lb_3.mean().item())
            print("lb_4: ", lb_4.mean().item())
            print("AL_1: ", AL_1.mean().item())
            print("ub_1: ", ub_1.mean().item())
            print("ub_2: ", ub_2.mean().item())
            print("ub_3: ", ub_3.mean().item())
            print("ub_4: ", ub_4.mean().item())
            print("AL_2: ", AL_2.mean().item())

        
        obj_fcn = problem.obj_fn(x).mean().item()

        # print("Objective function value: ", obj_fcn)
        
        # # diagnostics
        # s_plus = (s_full + muB).view(-1)                         # [B·m]
        # w_plus = (w_full + muB).view(-1)                         # [B·m]
        # d_val  = (c_val + s_full).view(-1)   # c(x)–s,  [B·m]

        # print(f"[dbg] s+μB min {s_plus.min():.3e} max {s_plus.max():.3e} | "
        #     f"w+μB min {w_plus.min():.3e} max {w_plus.max():.3e} | "
        #     f"c-s min {d_val.min():.3e} max {d_val.max():.3e}")

        chi = problem.chi(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB)
        if (device == 'cpu') & (print_level == 3):
            print(f"[IPM‐iter {outer}] μP={muP.item():.1e}, "
                f"μB={muB.mean().item():.1e}, M={M_k_new:.4e}, ∥r∥={chi.mean().item():.4e}")
            
            print("x: ", x.mean().item())
            print("z1: ", z1.mean().item())
            print("z2: ", z2.mean().item())

            for i in range(min(5, lb.shape[0])):
                print(f"obj[{i}] = {problem.obj_fn(x).view(B)[i]:.6f}")
        # print("s: ", s_flat.mean().item())
        # print("y: ", y_flat.mean().item())
        # print("w: ", w_flat.mean().item())

        #resid_c = c_val + s_full
        #y_col   = resid_c / muP
        #w_col   = muB / (s_col + muB) - y_col

        # obtaining each of the gradients
        grad_x, grad_s, grad_y, grad_vv, grad_z1, grad_z2, grad_w = problem.merit_grad_M(
            x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB, bad_x1, bad_x2, bad_z1, bad_z2
        )
        # flatten gradients
        if grad_x is not None: 
            grad_x = grad_x.view(B,   n)
        # if grad_x1 is not None:
        #     grad_x1 = grad_x1.view(B, n)
        # if grad_x2 is not None:
        #     grad_x1 = grad_x1.view(B, n)
        if grad_z1 is not None:
            grad_z1 = grad_z1.view(B, n)
        if grad_z2 is not None:
            grad_z2 = grad_z2.view(B, n)

        # if train_test != "train":
        #     inf_x = grad_x.abs().amax(dim=1)   # [B]
        #     # inf_x1 = grad_x1.abs().amax(dim=1)   # [B]
        #     # inf_x2 = grad_x2.abs().amax(dim=1)   # [B]
        #     inf_z1 = grad_z1.abs().amax(dim=1)   # [B]
        #     inf_z2 = grad_z2.abs().amax(dim=1)   # [B]
        #     if (inf_x < TOL).all() and (inf_z1 < TOL).all() and (inf_z2 < TOL).all(): # stopping criteria
        #         # val += 1
        #         # if (val_stop >= val):
        #         #     val_flag = True
        #         #     return val_flag
        #         # for val flag, need to return something otherwise also
        #         break


        # primal feasibility
        P = problem.primal_feasibility(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB)      
        # dual feasibility
        D = problem.dual_feasibility(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB)        
        # complementarity
        C = problem.complementarity(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB)    
        if (device == 'cpu') & (print_level == 3):
            print("Primal Feasability: ", P.mean())
            print("Dual Feasability: ", D.mean()) # stationarity is giving me issues
            for i in range(min(5, lb.shape[0])):
                print(f"Dual Feasability[{i}] = {D.view(B)[i]:.6f}")
            for i in range(min(5, lb.shape[0])):
                print(f"Complimentarity[{i}] = {C.view(B)[i]:.6f}")
            # for i in range(min(5, lb.shape[0])):
            #     print(f"Problem {i}:")
            #     for j in range(lb.shape[1]):
            #         print(f"  Variable {j}: z1 = {z1[i, j].item()}, z2 = {z2[i, j].item()}")
            #         print(f"obj_grad[{i}]", problem.obj_grad(x)[i])
            #     print()
            print("Complementarity: ", C.mean())
            if ((D.mean() < 0.1) & (P.mean() < 0.1) & (C.mean() < 0.1)):
                print("CONVERGED")
            thresh = 1e-2
            ok = (P < thresh) & (D < thresh) & (C < thresh)   # elementwise
            print("Percentage converged: ", sum(ok)/B)

        if train_test != "train":
            thresh = 1e-2
            ok = (P < thresh) & (D < thresh) & (C < thresh)   # elementwise
            if ok.all().item():   # True iff ALL problems pass
                break

        # 1) compute the two scaling norms ‒ D^P and D^B infinity‐norms
        #    D^P = μP I   →   ||D^P||_∞ = μP
        #Dp_norm = muP.view(-1)                         # [B] or scalar

        #    D^B = S_B W_B^{-1},  with S_B = s + μB,  W_B = w + μB
        #    so each diagonal entry is (s_i+μB)/(w_i+μB) →  ||D^B||_∞ = max_i |(s+μB)/(w+μB)|
        # print(x1.shape)
        # print(z1.shape)
        # print(muB.shape)
        DB = torch.cat([
            ((x1.squeeze(-1) + muB)/(z1.squeeze(-1) + muB)).abs().squeeze(-1),
            ((x2.squeeze(-1) + muB)/(z2.squeeze(-1) + muB)).abs().squeeze(-1)
        ], dim=1)         # [B, 2n]
        DB_norm = DB.amax(dim=1)  # [B]
        #print(DB_norm.mean())

        # #print("grade_x shape: ", grad_x_feas.shape)

        # # 2) check the four M‐iterate conditions (25a)–(25d):
        # #    ‖∇_x M‖_∞   ≤ τ_k
        # #    ‖∇_s M‖_∞   ≤ τ_k
        # #    ‖∇_y M‖_∞   ≤ τ_k · ‖D^P‖_∞
        # #    ‖∇_w M‖_∞   ≤ τ_k · ‖D^B‖_∞

        # # first compute the per‐instance infinity norms:
        # nx = grad_x.abs().amax(dim=1)   # [B]
        # nz1 = grad_z1.abs().amax(dim=1)   # [B]
        # nz2 = grad_z2.abs().amax(dim=1)   # [B]


        # # now form the boolean mask:
        # cond_M = (
        #     (nx <= tau_k)        &
        #     #(nx1 <= tau_k)      &    # primal‐slack test
        #     #(nx2 <= tau_k)      &    # primal‐slack test
        #     (nz1 <= tau_k * DB_norm) &
        #     (nz2 <= tau_k * DB_norm)
        # )     

        # print("nx: ", nx.mean().item(), " tau_k: ", tau_k)  
        # # print("nx1: ", nx1.mean().item(), " tau_k: ", tau_k)    
        # # print("nx2: ", nx2.mean().item(), " tau_k: ", tau_k)      
        # print("nz1: ", nz1.mean().item(), " tau_k * DB: ", (tau_k * DB_norm).mean().item()) 
        # print("nz1: ", nz2.mean().item(), " tau_k * DB: ", (tau_k * DB_norm).mean().item()) 
        # #print("nz2: ", nz2.mean().item(), " tau_k * Db_norm_2: ", (tau_k * Db_norm_2).mean().item())  

        Mtest = problem.Mtest(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB, M_max, bad_x1, bad_x2, bad_z1, bad_z2)   
        
        # print("Variables passing Mtest: ", (Mtest.abs() < 0.1).sum(dim = 0))

        # if (Mtest.abs() < 0.1).all():
        #     print("SUCCESS")
        #     return True 

        #if (M_k_new < (0.8*M_k)):
        
        # if train_test == "train":
        #     if (outer % 50 == 0):
        #         x1E[mask_M] = x1[mask_M].clamp(0.0, x1_max_e)
        #         x2E[mask_M] = x2[mask_M].clamp(0.0, x2_max_e)
        #         z1E[mask_M] = z1[mask_M].clamp(0.0, z1_max_e)
        #         z2E[mask_M] = z2[mask_M].clamp(0.0, z2_max_e)
        #         muB = muB/2

        mask_O = chi < chi_max               # [B]  O-iterate candidates
        mask_M = (~mask_O) & (Mtest < tolM)  # M if not O
        mask_F = ~(mask_O | mask_M)          # the rest

        mask_lb = torch.isfinite(lb_vals) if lb_vals is not None else None
        mask_ub = torch.isfinite(ub_vals)   if ub_vals is not None else None

        mask_lb = mask_lb.unsqueeze(-1)
        mask_ub = mask_ub.unsqueeze(-1)


        #--------------------------------------------------------------------
        # Allocate new “E” tensors from their current values
        #--------------------------------------------------------------------
        x1E_new, x2E_new = x1E.clone(), x2E.clone()     # [B,n,1]
        z1E_new, z2E_new = z1E.clone(), z2E.clone()     # [B,n,1]
        chi_max_new      = chi_max.clone()              # [B]
        tolM_new         = tolM.clone()                 # [B]
        muB_new          = muB.squeeze(-1).clone()                  # [B]
        muB = muB.squeeze(-1)

        if mask_O.any():
            #print("O-iterate")
            # 1) reduce chi_max by half
            chi_max_new[mask_O] *= 0.5
            #chi_max_new[mask_O] = chi[mask_O]

            # 2) project x into the box → slacks for those samples
            raw_x1 = (x - lb).clamp(min=eps_t)          # [B,n,1]
            raw_x2 = (ub - x).clamp(min=eps_t)
            if mask_lb is not None:
                raw_x1 = torch.where(mask_lb, raw_x1, torch.zeros_like(raw_x1))
            if mask_ub is not None:
                raw_x2 = torch.where(mask_ub, raw_x2, torch.zeros_like(raw_x2))

            x1E_new[mask_O] = torch.maximum(raw_x1[mask_O], torch.zeros_like(raw_x1[mask_O]))
            x2E_new[mask_O] = torch.maximum(raw_x2[mask_O], torch.zeros_like(raw_x2[mask_O]))
            z1E_new[mask_O] = z1[mask_O]
            z2E_new[mask_O] = z2[mask_O]

        if mask_M.any():
            #print("M-iterate")
            #1) clamp (inequality part) of slacks & multipliers
            x1E_new[mask_M] = x1[mask_M].clamp(0.0, x1_max_e)
            x2E_new[mask_M] = x2[mask_M].clamp(0.0, x2_max_e)
            z1E_new[mask_M] = z1[mask_M].clamp(0.0, z1_max_e)
            z2E_new[mask_M] = z2[mask_M].clamp(0.0, z2_max_e)

            #muB_new[(mask_M) & (muB_new > 1e-10)] = muB_new[(mask_M) & (muB_new > 1e-10)]/2
            #muB_new[~mask_M] = muB[~mask_M]
            #if train_test != "train":
                # # make tolM broadcastable to [B_M, 1, 1]
            tolM_M = tolM_new[mask_M].view(-1, 1, 1)          # [B_M,1,1]

            # 1) complementarity check (C already [B])
            bad_comp = C[mask_M] > tolM_new[mask_M]            # [B_M]

            # 2) slack & multiplier sign checks
            bad_x1 = (x1[mask_M] < -tolM_M).any(dim=2).any(dim=1)   # [B_M]
            bad_x2 = (x2[mask_M] < -tolM_M).any(dim=2).any(dim=1)
            bad_z1 = (z1[mask_M] < -tolM_M).any(dim=2).any(dim=1)
            bad_z2 = (z2[mask_M] < -tolM_M).any(dim=2).any(dim=1)

            # aggregate “bad” flag  (logical OR)
            shrink_muB = bad_comp | bad_x1 | bad_x2 | bad_z1 | bad_z2   # [B_M]

            # keep if NOT bad
            keep_muB  = ~shrink_muB                                     # [B_M]

            # apply update:  μB ← μB / μBfac   where shrink flag is True
            muB_new[mask_M] = torch.where(keep_muB,
                                        muB[mask_M],
                                        muB[mask_M] / 2)

            #3) halve tolM for M samples
            tolM_new[mask_M] = 0.5 * tolM[mask_M]

        tolM  = tolM.clone()        # ← breaks the broadcast view
        chi_max = chi_max.clone()
        muB  = muB.clone()

        # Commit updated values back to the solver state
        chi_max.copy_(chi_max_new)
        tolM.copy_(tolM_new)
        muB.copy_(muB_new)
        muB = muB.unsqueeze(-1)

        x1E.copy_(x1E_new)
        x2E.copy_(x2E_new)
        z1E.copy_(z1E_new)
        z2E.copy_(z2E_new)

        slack1 = x1.squeeze(-1) + muB        # should stay > 0
        slack2 = x2.squeeze(-1) + muB        # should stay > 0

        # 2) build a mask of infeasible entries
        bad_x1 = (slack1 <= 0)   # shape [B,n,1], True wherever x1+μB ≤ 0
        bad_x2 = (slack2 <= 0)   # shape [B,n,1], True wherever x2+μB ≤ 0

        total_infeas = int(bad_x1.sum().item())
        #print(f"Total infeasible variables wrt to x1 in batch: {total_infeas}")

        slackz1 = (z1.squeeze(-1) + muB)    # [B,n]
        slackz2 = (z2.squeeze(-1) + muB)    # [B,n]

        # build masks of the exact indices that went non-positive
        bad_z1 = (slackz1 <= 0).unsqueeze(-1)  # [B,n,1]
        bad_z2 = (slackz2 <= 0).unsqueeze(-1)  # [B,n,1]

        # Option A: out-of-place with torch.where
        z1 = torch.where(bad_z1, torch.zeros_like(z1), z1)
        z2 = torch.where(bad_z2, torch.zeros_like(z2), z2)


        # if (P <= tolP).all() and (D <= tolD).all() and (C <= tolC).all():
        #     break

        # if (P <= tau_k).all():
        #     # keep μP
        #     pass
        # else:
        #     muP = muP * 0.5

        # if (C <= tau_k).all() and (s_col >= -tau_k).all() and (w_col >= -tau_k).all():
        #     # keep μB
        #     pass
        # else:
        #     muB = muB * 0.5

        # simple barrier schedule
        # change this to make it more complicated as in projected search IPM
        # feas = resid_c.abs().max().item()
        # if feas < 0.1 * muP.item():
        #     muP *= 0.5
        #     muB *= 0.5
        # else:
        #     if (M_k_new > M_k):
        #         break
        # issue is sometimes it is ok for the merit function to increase, especially when we shrink the mu, so idea is kind of flawed. 

        # if (M_k_new > M_max): # this is just to ensure that the merit function does not explode, which it does initially. 
        #     # penalize for every step it does not take
        #     #total_inner_loss = total_inner_loss + (1e10 * (max_outer - outer))
        #     #flag = True
        #     break

        M_k = M_k_new
    ds_elapsed = time.perf_counter() - ds_start
    print("Time elapsed: ", ds_elapsed/B)
    print("Loss for this step: ", step_loss_vector.mean())
    print("Primal Feasability: ", P.mean())
    print("Dual Feasability: ", D.mean()) # stationarity is giving me issues
    print("Complementarity: ", C.mean())
    print("Number of Iterations = ", outer)
    thresh = 1e-2
    # ok = (P < thresh) & (D < thresh) & (C < thresh)   # elementwise
    # if ok.all().item():   # True iff ALL problems pass
    #     return True

    # meta_opt.zero_grad(set_to_none=True)      # clear old gradients
    # total_inner_loss.backward()                      # back-prop through the micro step
    # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    # torch.nn.utils.clip_grad_value_(net.parameters(), 10.0)
    # torch.autograd.set_detect_anomaly(True)
    # meta_opt.step()                           # update the LSTM weights
    
    
    #return False

    # if flag:
    #     iterations = max_outer
    # else:
    #     iterations = outer

    # return final (x,s,y,w) and average inner‐loop loss
    #return x_flat, s_flat, y_flat, w_flat, total_inner_loss / (outer)


# function to train
# def train_epoch(pool, net, meta_opt):
#     net.train()
#     meta_opt.zero_grad(set_to_none=True)
#     _, _, _, _, loss = solve_one_qcqp(pool, net) # one solve qcqp on the full batch
#     print(f"[Epoch]   loss = {loss.item():.6f}")
#     loss.backward()                        #   backward propogation
#     # clipping the gradients so that it does not explode?
#     torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
#     meta_opt.step()

#     return loss.item()                     


# @torch.no_grad()
# def evaluate(pool, net, tag='test'):
#     net.eval()

#     # run one solve qcqp to get the final solution
#     x_fin, s_fin, y_fin, w_fin, _ = solve_one_qcqp(pool, net)

#     # checking ∞‐norm feasibility: max_i |g_i(x) - s_i|
#     x_col = x_fin.unsqueeze(-1)      # [B,n,1]
#     s_col = s_fin.unsqueeze(-1)      # [B,m_tot,1]
#     feas_inf = pool.primal_feasibility(x_col, s_col).mean().item()
    
#     # primal objective f(x) = ½ x^T Q x + p^T x
#     #    obj_fn returns a [B,1,1] tensor
#     f_vals = pool.obj_fn(x_col).view(-1)    # → [B]
#     mean_obj = f_vals.mean().item()

#     print(f"{tag:>5} | feas∞ {feas_inf:.2e} | mean obj {mean_obj:.3e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--ckpt", default="l20_lstm_ipm.pt")
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # path to randomly generated data, will need to create this for convex QP functions at sometime 
    #mat_name = f"random_convex_qcqp_dataset_var{DIM_X}_ineq{DIM_S}_eq{DIM_S}_ex10000" # change the path
    # mat_name = "unconstrained_QP_convex_full_Q_dim_100"
    # file_path = os.path.join("datasets", "qp", f"{mat_name}.mat")
    # # make this part a loop so that user can choose

    # # train, validation, and test pools
    # train_pool = QP(prob_type='QP_unconstrained', learning_type='train', file_path=file_path, seed=args.seed)
    # val_pool   = QP(prob_type='QP_unconstrained', learning_type='val', file_path=file_path, seed=args.seed)
    # test_pool  = QP(prob_type='QP_unconstrained', learning_type='test', file_path=file_path, seed=args.seed)
    # # change the type of function

    # # PS_L20_LSTM signature is: __init__(self, n: int, m: int, hidden_size: int=256, num_layers: int=1)
    # net = PS_L20_LSTM(
    #     problem = train_pool, 
    #     #m_ineq=DIM_S,
    #     #m_eq=DIM_S,
    #     #hidden_size=HIDDEN,
    #     #num_layers=K_INNER,   
    #     K_inner = K_INNER,  
    #     device=DEVICE
    # ).to(DEVICE)

    # mat_names = [
    #     "unconstrained_QP_convex_full_Q_dim_10",
    #     "unconstrained_QP_convex_full_Q_dim_30",
    #     "unconstrained_QP_convex_full_Q_dim_50",
    #     "unconstrained_QP_convex_full_Q_dim_70",
    #     "unconstrained_QP_convex_full_Q_dim_90",
    # ]

    # mat_names = [
    #     "unconstrained_QP_nonconvex_full_Q_dim_10",
    #     "unconstrained_QP_nonconvex_full_Q_dim_30",
    #     "unconstrained_QP_nonconvex_full_Q_dim_50",
    #     "unconstrained_QP_nonconvex_full_Q_dim_70",
    #     "unconstrained_QP_nonconvex_full_Q_dim_90"
    # ]

    mat_names = [
        "unconstrained_QP_nonconvex_full_Q_dim_500"
    ]

    file_paths = [os.path.join("datasets", "qp", f"{nm}.mat") for nm in mat_names]

    # 2) build pools for each dataset
    pools = []
    for fp in file_paths:
        pools.append({
            "train": QP(prob_type='QP_unconstrained', learning_type='train', file_path=fp, seed=args.seed),
            "val":   QP(prob_type='QP_unconstrained', learning_type='val',   file_path=fp, seed=args.seed),
            "test":  QP(prob_type='QP_unconstrained', learning_type='test',  file_path=fp, seed=args.seed),
        })

    # 3) init net using the first train pool (shapes/masks assumed consistent with padding)
    net = PS_L20_LSTM(problem=pools[0]["train"], K_inner=K_INNER, device=DEVICE).to(DEVICE)

    # training loop
    if args.mode == "train":
        meta_opt = optim.Adam(net.parameters(), lr=LR_META) # adam optimizer
        # meta_opt = torch.optim.LBFGS(
        #     net.parameters(),
        #     lr=1.0,                 # step size for the line search; 1.0 is a good start
        #     max_iter=10,            # inner iterations per step
        #     history_size=50,        # number of curvature pairs to keep
        #     line_search_fn='strong_wolfe',  # more stable than default backtracking
        #     tolerance_grad=1e-7,
        #     tolerance_change=1e-9
        # )
        import copy
        import numpy as np

        BATCH = 200
        #print("N: ", N)
        for ep in range(1, EPOCHS+1):
            #print("No chi")
            print("EPOCH: ", ep)
            ds_order = torch.randperm(len(pools)).tolist()

            for di, ds_idx in enumerate(ds_order, 1):
                train_pool = pools[ds_idx]["train"]
                print(f"  DATASET {di}/{len(pools)}: {mat_names[ds_idx]}")
                t0 = time.time()
                N  = train_pool.Q.shape[0]
                perm = torch.arange(N, device=train_pool.Q.device)
                for i in range(0, N, BATCH):
                    batch_idx = i // BATCH + 1
                    num_batches = (N + BATCH - 1) // BATCH  # ceil div without math
                    print(f"BATCH: {batch_idx}/{num_batches}")
                    if (DEVICE == 'cpu'):
                        idxs = perm[i:i+BATCH]            # torch.LongTensor of size ≤128
                        idxs_list = idxs.tolist()                 # for slicing Python lists

                        # shallow‐copy the pool
                        mini = copy.copy(train_pool)

                        # walk through every attribute in train_pool
                        for name, val in train_pool.__dict__.items():
                            # 1) torch.Tensor with first dim == N  → slice it
                            if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.size(0) == N:
                                setattr(mini, name, val[idxs])

                            # 2) numpy array with first dim == N  → slice it
                            elif isinstance(val, np.ndarray) and val.shape and val.shape[0] == N:
                                setattr(mini, name, val[idxs.cpu().numpy()])

                            # 3) Python list of length N → pick indices
                            elif isinstance(val, list) and len(val) == N:
                                setattr(mini, name, [val[i] for i in idxs_list])
                    else:
                        # assume: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        idxs      = perm[i : i + BATCH]              # LongTensor on default device
                        idxs      = idxs.to(DEVICE).long()           # ensure it’s on GPU
                        idxs_list = idxs.cpu().tolist()              # for Python‐list indexing

                        # shallow‐copy the pool
                        mini = copy.copy(train_pool)

                        for name, val in train_pool.__dict__.items():
                            # 1) PyTorch tensor → GPU‐slice directly
                            if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.size(0) == N:
                                setattr(mini, name, val[idxs])

                            # 2) NumPy array → convert once to GPU tensor, then slice
                            elif isinstance(val, np.ndarray) and val.shape and val.shape[0] == N:
                                tval = torch.from_numpy(val).to(DEVICE)
                                setattr(mini, name, tval[idxs])

                            # 3) Python list → use saved Python indices
                            elif isinstance(val, list) and len(val) == N:
                                setattr(mini, name, [val[j] for j in idxs_list])

                    flag = solve_one_qcqp(meta_opt, mini, net)
                    # get the objective 
                    # evaluate(val_pool, net, tag="val")
                print(f"Epoch {ep:3d} | {time.time()-t0:.1f}s")
                if flag:
                    break
            #solve_one_qcqp(meta_opt, val_pool, net, train_test = 'val')
        torch.save(net.state_dict(), args.ckpt)
        print(f"✓ weights saved to {args.ckpt}")

    # test evaluation 
    else:
        import time
        total_start = time.perf_counter()
        net.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
        print(f"✓ loaded weights from {args.ckpt}")
        net.to(DEVICE).eval()
        with torch.inference_mode():              # stronger than no_grad
            torch.backends.cudnn.benchmark = True
            for ds_idx, pack in enumerate(pools):
                print(f"\nTesting: {mat_names[ds_idx]}")
                ds_start = time.perf_counter()
                _ = solve_one_qcqp(optim.Adam(net.parameters(), lr=LR_META), pack["test"], net, train_test='test')
                # IMPORTANT for accurate GPU timing:
        
                if torch.cuda.is_available() and DEVICE == "cuda":
                    torch.cuda.synchronize()

                ds_elapsed = time.perf_counter() - ds_start
                print(f"[{mat_names[ds_idx]}] elapsed: {ds_elapsed:.3f}s")

                # def run_ipopt_box(problem, x0=None, tol=1e-8, max_iter=500):

                #     # assuming batch size is 1, might have to change that
                    
                #     import numpy as np
                #     """
                #     Returns: x_opt, obj, runtime_seconds, ipopt_iters
                #     """
                #     B = problem.Q.shape[0]
                #     n_vec = problem.n_vec
                #     n, m_ineq, m_eq = problem.num_var, problem.num_ineq, problem.num_eq
                #     device = DEVICE
                #     Q = np.asarray(problem.Q, float).reshape(n, n)
                #     print(Q.shape)
                #     p = np.asarray(problem.p, float).reshape(n)
                #     lb = np.asarray(problem.lb, float).reshape(n)
                #     ub = np.asarray(problem.ub, float).reshape(n)

                #     if x0 is None:
                #         # simple feasible start: midpoint where finite, else 0 with small nudges
                #         lb_vals = problem.lb.expand(B, n).to(device)    # [B,n]
                #         ub_vals = problem.ub.expand(B, n).to(device)    # [B,n]

                #         # masks for which bounds are finite
                #         has_lb = torch.isfinite(lb_vals).to(device)                    # [B,n]
                #         has_ub = torch.isfinite(ub_vals).to(device)                    # [B,n]

                #         # start with zeros
                #         x_init = torch.zeros((B, n), device=device)       # [B,n]

                #         # both bounds → midpoint
                #         both = has_lb & has_ub
                #         #both = both.to(device)
                #         x_init[both] = 0.5 * (lb_vals[both] + ub_vals[both])

                #         # lb only → lb + 1
                #         lb_only = has_lb & ~has_ub 
                #         x_init[lb_only] = lb_vals[lb_only] + 1.0

                #         # ub only → ub − 1
                #         ub_only = has_ub & ~has_lb
                #         x_init[ub_only] = ub_vals[ub_only] - 1.0
                #         x0 = x_init
                #         x0 = x0.reshape(n)
                    
                #     print(x0.shape)

                #     nlp = BoxQP(Q, p, lb, ub, tol=tol, max_iter=max_iter)

                #     t0 = time.perf_counter()
                #     x_opt, info = nlp.solve(x0)
                #     runtime = (time.perf_counter() - t0)

                #     obj = 0.5 * x_opt.dot(Q.dot(x_opt)) + p.dot(x_opt)
                #     iters = nlp.iters
                #     return x_opt, obj, runtime, iters
                # x_opt, obj, t, iters = run_ipopt_box(pack["test"], tol=1e-3, max_iter=500)
                # print(f"IPOPT: iters={iters}, time={t:.3f}s, obj={obj:.6f}")
                sols, _, para_times, iters = pack["test"].opt_solve(solver_type='ipopt_box_qp', tol=1e-3)
                best_obj = pack["test"].obj_fn(torch.tensor(sols).unsqueeze(-1).float()).mean()
                print('Best objective value:', best_obj)
                print('Original Solver Time: {}'.format(para_times))
                print('Original Iters:', iters)


    #evaluate(test_pool, net, tag="test")

