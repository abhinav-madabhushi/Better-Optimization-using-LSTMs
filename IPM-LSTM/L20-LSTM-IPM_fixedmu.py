import argparse, time, torch, torch.optim as optim
import torch.nn.init as init

import os
from problems.Convex_QCQP import Convex_QCQP        
from problems.QP import QP
from models.LSTM_L20_Projected_Math import PS_L20_LSTM         

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

DIM_X    = 2        # decision variables
DIM_S    = 50         # inequality slacks
HIDDEN   = 128        # LSTM hidden size, did hidden size 32 make the model better? can play around with this
K_INNER  = 1         # micro-steps per outer iteration, need to fix this in the code

EPOCHS   = 1000                    # meta-training epochs
LR_META  = 1e-4                   # learning-rate for Adam

MAX_OUTER  = 100
tolP, tolD, tolC = 1e-4, 1e-4, 1e-4
TOL = 1e-4
S_MAX_E = 10  
Y_MAX_E = 10   
W_MAX_E = 10 

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
        #wall = -muB.unsqueeze(-1).expand_as(v_k)
        wall = torch.minimum(v_k - sigma * (v_k - ell) , torch.full_like(v_k, 0))
        lower = wall + eps
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


def solve_one_qcqp(meta_opt, problem: "Convex_QCQP",
                   net: PS_L20_LSTM,
                   muP0=1e-4, muB0=1e-1,
                   max_outer=MAX_OUTER, train_test = 'train', val_stop = 5, **kwargs):
    
    # change muB to 1e-2: done
    """
    Projected-search IPM loop powered by a diagonal L20-LSTM.
    Returns final (x,s,y,w) and the average inner loss.
    """

    # loop through batches here

    if train_test == 'train':
        print("train")
    else: 
        print("test")

    B = problem.Q.shape[0]
    n, m_ineq, m_eq = problem.num_var, problem.num_ineq, problem.num_eq
    device = DEVICE

    val = 0

    # initializing the variables, will need initialize v also here later 
    if m_eq > 0:
        A_pinv = torch.linalg.pinv(problem.A)       # [B,n,p]
        x  = torch.bmm(A_pinv, problem.b)        # [B,n,1]
    else:
        x  = torch.zeros(B, n, 1, device=device)
        v = None
        vE = None

    # slack variable s initialization for the inequalities
    if m_ineq > 0:
        g0    = problem.ineq_resid(x)           # [B,m_ineq,1]
        s = torch.clamp(-g0, min=1e-6)           # [B,m_ineq,1] 
        muP = x.new_tensor([muP0])   
    else:
        s = None
        y = None
        w = None
        sE = None
        yE = None
        wE = None

    eps = 1e-6

    # initialize the duals anbd the shifts            
    muB = x.new_tensor([muB0])
    muP = x.new_tensor([muP0])
    tau_k = 2
    tolM = torch.tensor(TOLM)

    # initializing x using the bounds
    if problem.num_lb != 0 and problem.num_ub != 0:
        # initialize x
        lb_vals = problem.lb.expand(B, n)    # [B,n]
        ub_vals = problem.ub.expand(B, n)    # [B,n]

        # masks for which bounds are finite
        has_lb = torch.isfinite(lb_vals)                  # [B,n]
        has_ub = torch.isfinite(ub_vals)                  # [B,n]

        # start with zeros
        x_init = torch.zeros((B, n), device=device)       # [B,n]

        # both bounds → midpoint
        both = has_lb & has_ub
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
        lb = problem.lb.view(B, n, 1).expand(B, n, 1)   # [B,n,1]
        ub = problem.ub.view(B, n, 1).expand(B, n, 1)   # [B,n,1]

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
        # dual   z1_j = μB / x1_j
        z1     = torch.where(mask_lb,
                            muB / x1,
                            torch.zeros_like(x1))

        # --- upper‐bound slack & dual ---
        # slack  x2_j = clamp(u_j - x_j, min=eps)
        raw_x2 = (ub - x).clamp(min=eps_t)               # [B,n,1]
        x2     = torch.where(mask_ub, raw_x2,
                            torch.zeros_like(raw_x2))
        # dual   z2_j = μB / x2_j
        z2     = torch.where(mask_ub,
                            muB / x2,
                            torch.zeros_like(x2))
        
        x1E, x2E, z1E, z2E = x1.clone(), x2.clone(), z1.clone(), z2.clone()
        x1_max_e = torch.tensor(X1_MAX_E, device=device)
        x2_max_e = torch.tensor(X2_MAX_E, device=device)
        z1_max_e = torch.tensor(Z1_MAX_E, device=device)
        z2_max_e = torch.tensor(Z2_MAX_E, device=device)


    if train_test == "train":
        muB_vals = 0.1 * 0.5 ** torch.arange(15, device=device, dtype=torch.float32)
        M        = muB_vals.numel()

        def expand(t): return t.repeat_interleave(M, 0)

        # replicate blocks
        x,x1,x2,z1,z2   = map(expand, (x,x1,x2,z1,z2))
        x1E,x2E,z1E,z2E  = map(expand, (x1E,x2E,z1E,z2E))

        muB = muB_vals.repeat(B).unsqueeze(1)  # [B*M,1]

        problem.expand_all_matrices(15)
    else:
        muB = muB.expand(B, 1)

    B = problem.Q.shape[0]

    lb = problem.lb.view(B, n, 1).expand(B, n, 1)   # [B,n,1]
    ub = problem.ub.view(B, n, 1).expand(B, n, 1)   # [B,n,1]

    lb_vals = problem.lb.expand(B, n)    # [B,n]
    ub_vals = problem.ub.expand(B, n)    # [B,n]
    

    

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

    #muB = muB.expand(B, 1)
    tolM = tolM.expand(B)


    h = torch.empty(net.lstm.num_layers, B, net.lstm.hidden_size, device=device)
    c = torch.empty_like(h)

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
    
    # r_k includes μP, μB
    r_k = v0.clone()         

    slack1 = x1.squeeze(-1) + muB        # should stay > 0
    slack2 = x2.squeeze(-1) + muB        # should stay > 0

    # 2) build a mask of infeasible entries
    bad_x1 = (slack1 <= 0)   # shape [B,n,1], True wherever x1+μB ≤ 0
    bad_x2 = (slack2 <= 0)   # shape [B,n,1], True wherever x2+μB ≤ 0

    total_infeas = int(bad_x1.sum().item())
    print(f"Total infeasible variables wrt to x1 in batch: {total_infeas}")

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

    chi_max = problem.chi(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB)
    print(f"[IPM‐iter {0}] μP={muP.item():.1e}, "
               f"μB={muB.mean().item():.1e}, M={M_k:.4e}, ∥r∥={chi_max.mean().item():.4e}")
    
    for i in range(5):
        print(f"obj[{i}] = {problem.obj_fn(x).squeeze(-1).squeeze(-1)[i]:.6f}")

    for i in range(min(5, lb.shape[0])):
        print(f"Problem {i}:")
        for j in range(lb.shape[1]):
            print(f"  Variable {j}: lower = {lb[i, j]}, upper = {ub[i, j]}")
        print()

    # NEED TO ADD DIAGNOSTICS FOR X1, X2, Z1, Z2
    
    # grad_x, grad_s_full, grad_y_full, grad_w_full = problem.merit_grad_M(
    #         x_col, s_full, y_full, w_full, sE, yE, wE, muP, muB
    #     )
    
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

    if train_test != "train":
        max_outer = max_outer * 2
    # outer IPM iterations
    for outer in range(1, max_outer+1):

        
        # flatten for LSTM call
        x = x.squeeze(-1)   
        x1 = x1.squeeze(-1)   
        x2= x2.squeeze(-1)
        z1 = z1.squeeze(-1)
        z2 = z2.squeeze(-1)


        # currently does 1 micro-iteration of the LSTM, can change that 
        (x, x1, x2, s, y, v, z1, z2, w), step_loss_vector, (lstm_states, r_k) = net(
            problem,
            x, x1, x2, s, y, v, z1, z2, w,
            x1E.squeeze(-1), x2E.squeeze(-1), sE, yE, vE, z1E.squeeze(-1), z2E.squeeze(-1), wE,
            muP, muB, r_k,
            lstm_states, bad_x1, bad_x2, bad_z1, bad_z2, M_max, 
            project_step=project_tube_all
        )

        # for each problem use diff model based on its value of muB, will have to code that in here. 

        step_loss = step_loss_vector.sum()

        print("Loss for this step: ", step_loss)

        x1E = x1E.detach(); x2E = x2E.detach();z1E = z1E.detach();z2E = z2E.detach()
        #sE = sE.detach(); yE = yE.detach(); wE = wE.detach(); vE = vE.detach()
        muB = muB.detach(); muP = muP.detach()

        x = x.detach()
        x1 = x1.detach()
        x2 = x2.detach()
        z1 = z1.detach()
        z2 = z2.detach()
        # y = y.detach()
        # w = w.detach()
        # v = v.detach()
        # s = s.detach()

        if train_test == "train":
            if outer % 20 == 0:
                meta_opt.zero_grad(set_to_none=True)      # clear old gradients
                step_loss.backward()                      # back-prop through the micro step
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                meta_opt.step()                           # update the LSTM weights

                h, c = lstm_states
                lstm_states = (h.detach(), c.detach())
                r_k = r_k.detach()


        # if train_test == 'train':
        #     meta_opt.zero_grad(set_to_none=True)      # clear old gradients
        #     step_loss.backward()                      # back-prop through the micro step
        #     torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        #     meta_opt.step()                           # update the LSTM weights
        # else:
        #     val_flag = False

        # reshape back to columns
        x = x.unsqueeze(-1)   
        x1 = x1.unsqueeze(-1)   
        x2= x2.unsqueeze(-1)
        z1 = z1.unsqueeze(-1)
        z2 = z2.unsqueeze(-1)

        # display merit/residual
        M_k_new = problem.merit_M(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB, bad_x1, bad_x2, bad_z1, bad_z2).mean().item()
        
        f_val, lb_1, lb_2, lb_3, lb_4, ub_1, ub_2, ub_3, ub_4 = problem.merit_M_indi(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB, bad_x1, bad_x2, bad_z1, bad_z2)
        print("f_val: ", f_val.mean().item())
        print("lb_1: ", lb_1.mean().item())
        print("lb_2: ", lb_2.mean().item())
        print("lb_3: ", lb_3.mean().item())
        print("lb_4: ", lb_4.mean().item())
        print("ub_1: ", ub_1.mean().item())
        print("ub_2: ", ub_2.mean().item())
        print("ub_3: ", ub_3.mean().item())
        print("ub_4: ", ub_4.mean().item())


        total_inner_loss = total_inner_loss + step_loss
        
        obj_fcn = problem.obj_fn(x).mean().item()

        print("Objective function value: ", obj_fcn)
        
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
        print(f"[IPM‐iter {outer}] μP={muP.item():.1e}, "
               f"μB={muB.mean().item():.1e}, M={M_k_new:.4e}, ∥r∥={chi.mean().item():.4e}")
        
        print("x: ", x.mean().item())
        print("z1: ", z1.mean().item())
        print("z2: ", z2.mean().item())

        for i in range(5):
            print(f"obj[{i}] = {problem.obj_fn(x).squeeze(-1).squeeze(-1)[i]:.6f}")
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

        if train_test != "train":
            inf_x = grad_x.abs().amax(dim=1)   # [B]
            # inf_x1 = grad_x1.abs().amax(dim=1)   # [B]
            # inf_x2 = grad_x2.abs().amax(dim=1)   # [B]
            inf_z1 = grad_z1.abs().amax(dim=1)   # [B]
            inf_z2 = grad_z2.abs().amax(dim=1)   # [B]
            if (inf_x < TOL).all() and (inf_z1 < TOL).all() and (inf_z2 < TOL).all(): # stopping criteria
                # val += 1
                # if (val_stop >= val):
                #     val_flag = True
                #     return val_flag
                # for val flag, need to return something otherwise also
                break


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
        print("Primal Feasability: ", P.mean())
        print("Dual Feasability: ", D.mean()) # stationarity is giving me issues
        for i in range(5):
            print(f"Dual Feasability[{i}] = {D.squeeze(-1).squeeze(-1)[i]:.6f}")
        for i in range(min(5, lb.shape[0])):
            print(f"Problem {i}:")
            for j in range(lb.shape[1]):
                print(f"  Variable {j}: z1 = {z1[i, j].item()}, z2 = {z2[i, j].item()}")
                print(f"obj_grad[{i}]", problem.obj_grad(x)[i])
            print()
        print("Complementarity: ", C.mean())

        # 1) compute the two scaling norms ‒ D^P and D^B infinity‐norms
        #    D^P = μP I   →   ||D^P||_∞ = μP
        #Dp_norm = muP.view(-1)                         # [B] or scalar

        #    D^B = S_B W_B^{-1},  with S_B = s + μB,  W_B = w + μB
        #    so each diagonal entry is (s_i+μB)/(w_i+μB) →  ||D^B||_∞ = max_i |(s+μB)/(w+μB)|
        print(x1.shape)
        print(z1.shape)
        print(muB.shape)
        DB = torch.cat([
            ((x1.squeeze(-1) + muB)/(z1.squeeze(-1) + muB)).abs().squeeze(-1),
            ((x2.squeeze(-1) + muB)/(z2.squeeze(-1) + muB)).abs().squeeze(-1)
        ], dim=1)         # [B, 2n]
        DB_norm = DB.amax(dim=1)  # [B]
        print(DB_norm.mean())

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
        
        # # print("Variables passing Mtest: ", (Mtest.abs() < 0.1).sum(dim = 0))

        # # if (Mtest.abs() < 0.1).all():
        # #     print("SUCCESS")
        # #     return True 

        # #if (M_k_new < (0.8*M_k)):
        # #muB = 0.5*muB

        # # if (outer % 10 == 0):
        # #     raw_x1 = (x - lb).clamp(min=eps_t)               # [B,n,1]
        # #     x1E     = torch.where(mask_lb, raw_x1,
        # #                         torch.zeros_like(raw_x1))
        # #     raw_x2 = (ub - x).clamp(min=eps_t)               # [B,n,1]
        # #     x2E     = torch.where(mask_ub, raw_x2,
        # #                         torch.zeros_like(raw_x2))
        # #     z1E = z1.clone()
        # #     z2E = z2.clone()
        # #     muB = 0.5*muB

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
            print("O-iterate")
            # 1) reduce chi_max by half
            chi_max_new[mask_O] *= 0.5

            # 2) project x into the box → slacks for those samples
            raw_x1 = (x - lb).clamp(min=eps_t)          # [B,n,1]
            raw_x2 = (ub - x).clamp(min=eps_t)
            if mask_lb is not None:
                raw_x1 = torch.where(mask_lb, raw_x1, torch.zeros_like(raw_x1))
            if mask_ub is not None:
                raw_x2 = torch.where(mask_ub, raw_x2, torch.zeros_like(raw_x2))

            x1E_new[mask_O] = raw_x1[mask_O]
            x2E_new[mask_O] = raw_x2[mask_O]
            z1E_new[mask_O] = z1[mask_O]
            z2E_new[mask_O] = z2[mask_O]

        if (train_test != 'train'):
            if mask_M.any():
                print("M-iterate")
                # 1) clamp (inequality part) of slacks & multipliers
                x1E_new[mask_M] = x1[mask_M].clamp(0.0, x1_max_e)
                x2E_new[mask_M] = x2[mask_M].clamp(0.0, x2_max_e)
                z1E_new[mask_M] = z1[mask_M].clamp(0.0, z1_max_e)
                z2E_new[mask_M] = z2[mask_M].clamp(0.0, z2_max_e)

                # make tolM broadcastable to [B_M, 1, 1]
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

                # 3) halve tolM for M samples
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
        print(f"Total infeasible variables wrt to x1 in batch: {total_infeas}")

        slackz1 = (z1.squeeze(-1) + muB)    # [B,n]
        slackz2 = (z2.squeeze(-1) + muB)    # [B,n]

        # build masks of the exact indices that went non-positive
        bad_z1 = (slackz1 <= 0).unsqueeze(-1)  # [B,n,1]
        bad_z2 = (slackz2 <= 0).unsqueeze(-1)  # [B,n,1]

        # Option A: out-of-place with torch.where
        z1 = torch.where(bad_z1, torch.zeros_like(z1), z1)
        z2 = torch.where(bad_z2, torch.zeros_like(z2), z2)

        slack1 = x1.squeeze(-1) + muB        # should stay > 0
        slack2 = x2.squeeze(-1) + muB        # should stay > 0
        
        # 2) build a mask of infeasible entries
        bad_x1 = (slack1 <= 0)   # shape [B,n,1], True wherever x1+μB ≤ 0
        bad_x2 = (slack2 <= 0)   # shape [B,n,1], True wherever x2+μB ≤ 0

        total_infeas = int(bad_x1.sum().item())
        print(f"Total infeasible variables wrt to x1 in batch: {total_infeas}")

        slackz1 = (z1.squeeze(-1) + muB)    # [B,n]
        slackz2 = (z2.squeeze(-1) + muB)    # [B,n]

        # build masks of the exact indices that went non-positive
        bad_z1 = (slackz1 <= 0).unsqueeze(-1)  # [B,n,1]
        bad_z2 = (slackz2 <= 0).unsqueeze(-1)  # [B,n,1]

        # Option A: out-of-place with torch.where
        z1 = torch.where(bad_z1, torch.zeros_like(z1), z1)
        z2 = torch.where(bad_z2, torch.zeros_like(z2), z2)
        

        M_k = M_k_new
    # if train_test == "train": 
    #     meta_opt.zero_grad(set_to_none=True)      # clear old gradients
    #     total_inner_loss.backward()                      # back-prop through the micro step
    #     torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    #     meta_opt.step()                           # update the LSTM weights


# def solve_one_qcqp_test(meta_opt, problem: "Convex_QCQP",
#                    models,
#                    muP0=1e-4, muB0=1e-1,
#                    max_outer=MAX_OUTER, train_test = 'test', val_stop = 5, **kwargs):
    
#     # change muB to 1e-2: done
#     """
#     Projected-search IPM loop powered by a diagonal L20-LSTM.
#     Returns final (x,s,y,w) and the average inner loss.
#     """

#     # loop through batches here

#     if train_test == 'train':
#         print("train")
#     else: 
#         print("test")

#     B = problem.Q.shape[0]
#     n, m_ineq, m_eq = problem.num_var, problem.num_ineq, problem.num_eq
#     device = DEVICE

#     val = 0

#     # initializing the variables, will need initialize v also here later 
#     if m_eq > 0:
#         A_pinv = torch.linalg.pinv(problem.A)       # [B,n,p]
#         x  = torch.bmm(A_pinv, problem.b)        # [B,n,1]
#     else:
#         x  = torch.zeros(B, n, 1, device=device)
#         v = None
#         vE = None

#     # slack variable s initialization for the inequalities
#     if m_ineq > 0:
#         g0    = problem.ineq_resid(x)           # [B,m_ineq,1]
#         s = torch.clamp(-g0, min=1e-6)           # [B,m_ineq,1] 
#         muP = x.new_tensor([muP0])   
#     else:
#         s = None
#         y = None
#         w = None
#         sE = None
#         yE = None
#         wE = None

#     eps = 1e-6

#     # initialize the duals anbd the shifts            
#     muB = x.new_tensor([muB0])
#     muP = x.new_tensor([muP0])
#     tau_k = 2
#     tolM = torch.tensor(TOLM)

#     # initializing x using the bounds
#     if problem.num_lb != 0 and problem.num_ub != 0:
#         # initialize x
#         lb_vals = problem.lb.expand(B, n)    # [B,n]
#         ub_vals = problem.ub.expand(B, n)    # [B,n]

#         # masks for which bounds are finite
#         has_lb = torch.isfinite(lb_vals)                  # [B,n]
#         has_ub = torch.isfinite(ub_vals)                  # [B,n]

#         # start with zeros
#         x_init = torch.zeros((B, n), device=device)       # [B,n]

#         # both bounds → midpoint
#         both = has_lb & has_ub
#         x_init[both] = 0.5 * (lb_vals[both] + ub_vals[both])

#         # lb only → lb + 1
#         lb_only = has_lb & ~has_ub
#         x_init[lb_only] = lb_vals[lb_only] + 1.0

#         # ub only → ub − 1
#         ub_only = has_ub & ~has_lb
#         x_init[ub_only] = ub_vals[ub_only] - 1.0
#         x = x_init.unsqueeze(-1)
#         xE = x.clone()

#         # initialize x1, x2, z1, and z2
#         lb = problem.lb.view(B, n, 1).expand(B, n, 1)   # [B,n,1]
#         ub = problem.ub.view(B, n, 1).expand(B, n, 1)   # [B,n,1]

#         # masks for where bounds are finite
#         mask_lb = torch.isfinite(lb)   # [B,n,1], True where ℓ_j > -∞
#         mask_ub = torch.isfinite(ub)   # [B,n,1], True where u_j < +∞

#         # floor tensor
#         eps_t   = torch.tensor(eps, device=device)

#         # --- lower‐bound slack & dual ---
#         # slack  x1_j = clamp(x_j - ℓ_j, min=eps)
#         raw_x1 = (x - lb).clamp(min=eps_t)               # [B,n,1]
#         x1     = torch.where(mask_lb, raw_x1,
#                             torch.zeros_like(raw_x1))
#         # dual   z1_j = μB / x1_j
#         z1     = torch.where(mask_lb,
#                             muB / x1,
#                             torch.zeros_like(x1))

#         # --- upper‐bound slack & dual ---
#         # slack  x2_j = clamp(u_j - x_j, min=eps)
#         raw_x2 = (ub - x).clamp(min=eps_t)               # [B,n,1]
#         x2     = torch.where(mask_ub, raw_x2,
#                             torch.zeros_like(raw_x2))
#         # dual   z2_j = μB / x2_j
#         z2     = torch.where(mask_ub,
#                             muB / x2,
#                             torch.zeros_like(x2))
        
#         x1E, x2E, z1E, z2E = x1.clone(), x2.clone(), z1.clone(), z2.clone()
#         x1_max_e = torch.tensor(X1_MAX_E, device=device)
#         x2_max_e = torch.tensor(X2_MAX_E, device=device)
#         z1_max_e = torch.tensor(Z1_MAX_E, device=device)
#         z2_max_e = torch.tensor(Z2_MAX_E, device=device)
    

#     # # building c(x) = [g(x); A x - b] : [B,m_tot,1]
#     # if problem.num_ineq>0:
#     #     g_col = problem.ineq_resid(x_col)                  # [B,m_ineq,1]
#     # else:
#     #     g_col = torch.zeros((B,0,1), device=device)
#     # if problem.num_eq>0:
#     #     eq_col = torch.bmm(problem.A, x_col) - problem.b      # [B,m_eq,1]
#     # else:
#     #     eq_col = torch.zeros((B,0,1), device=device)
#     # c_val = torch.cat([g_col, eq_col], dim=1)           # [B,m_tot,1]

#     # resid_c = (c_val + s_full)     # [B,m_tot,1]
#     # y_full   = resid_c / muP                         # [B,m_tot,1]
#     # w_full   = muB / (s_full + muB) - y_full           # [B,m_tot,1]

#     # sE, yE, wE = s_full.clone(), y_full.clone(), w_full.clone()
#     # s_max_e = torch.tensor(S_MAX_E, device=device)
#     # y_max_e = torch.tensor(Y_MAX_E, device=device)
#     # w_max_e = torch.tensor(W_MAX_E, device=device)

#     muB = muB.expand(B, 1)
#     tolM = tolM.expand(B)


#     h = torch.empty(net.lstm.num_layers, B, net.lstm.hidden_size, device=device)
#     c = torch.empty_like(h)

#     # Xavier‐uniform initialization for the hidden and cell state of LSTM
#     init.xavier_uniform_(h)
#     init.xavier_uniform_(c)
#     lstm_states = (h, c)

#     N = 0
#     parts = []

#     # helper to squeeze and register a block
#     def _add_block(var, name):
#         nonlocal N
#         if var is not None:
#             flat = var.squeeze(-1)       # [B, dim]
#             parts.append(flat)
#             N = N + flat.shape[1]           # accumulate width

#     # always include x
#     _add_block(x,  'x')

#     # slack/dual for bounds
#     # _add_block(x1, 'x1')   # lower‐slack
#     # _add_block(x2, 'x2')   # upper‐slack
#     _add_block(z1, 'z1')   # lower‐dual
#     _add_block(z2, 'z2')   # upper‐dual

#     # inequality slacks/duals
#     _add_block(s,  's')    # ineq‐slack
#     _add_block(w,  'w')    # ineq‐dual
#     _add_block(y,  'y')    # ineq‐multiplier

#     # equality multiplier
#     _add_block(v,  'v')    # eq‐multiplier

#     # creating v0 according to the math
#     v0 = torch.cat(parts, dim=1)    
    
#     # r_k includes μP, μB
#     r_k = v0.clone()         

#     slack1 = x1.squeeze(-1) + muB        # should stay > 0
#     slack2 = x2.squeeze(-1) + muB        # should stay > 0

#     # 2) build a mask of infeasible entries
#     bad_x1 = (slack1 <= 0)   # shape [B,n,1], True wherever x1+μB ≤ 0
#     bad_x2 = (slack2 <= 0)   # shape [B,n,1], True wherever x2+μB ≤ 0

#     total_infeas = int(bad_x1.sum().item())
#     print(f"Total infeasible variables wrt to x1 in batch: {total_infeas}")

#     slackz1 = (z1.squeeze(-1) + muB)    # [B,n]
#     slackz2 = (z2.squeeze(-1) + muB)    # [B,n]

#     # build masks of the exact indices that went non-positive
#     bad_z1 = (slackz1 <= 0).unsqueeze(-1)  # [B,n,1]
#     bad_z2 = (slackz2 <= 0).unsqueeze(-1)  # [B,n,1]

#     # Option A: out-of-place with torch.where
#     z1 = torch.where(bad_z1, torch.zeros_like(z1), z1)
#     z2 = torch.where(bad_z2, torch.zeros_like(z2), z2)             

#     total_inner_loss = 0.0

#     # display merit/residual
#     M_k = problem.merit_M(x, x1, x2, s,      
#                 y, v, z1, z2, w,    
#                 x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
#                 muP, muB, bad_x1, bad_x2, bad_z1, bad_z2).mean()
    
#     M_max = problem.merit_M(x, x1, x2, s,      
#                 y, v, z1, z2, w,    
#                 x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
#                 muP, muB, bad_x1, bad_x2, bad_z1, bad_z2)
    
#     #total_inner_loss = total_inner_loss + M_k

#     # print("x: ", x0_flat.mean().item())
#     # print("s: ", s0_flat.mean().item())
#     # print("y: ", y0_flat.mean().item())
#     # print("w: ", w0_flat.mean().item())
    

#     obj_fcn = problem.obj_fn(x).mean().item()

#     #print("Objective function value: ", obj_fcn)
        
#     # # diagnostics
#     # s_plus = (s + muB).view(-1)                         # [B·m]
#     # w_plus = (w + muB).view(-1)                         # [B·m]
#     # d_val  = (c_val + s_full).view(-1)   # c(x)–s,  [B·m]

#     # print(f"[dbg] s+μB min {s_plus.min():.3e} max {s_plus.max():.3e} | "
#     #     f"w+μB min {w_plus.min():.3e} max {w_plus.max():.3e} | "
#     #     f"c-s min {d_val.min():.3e} max {d_val.max():.3e}")

#     chi_max = problem.chi(x, x1, x2, s,      
#                 y, v, z1, z2, w,    
#                 x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
#                 muP, muB)
#     print(f"[IPM‐iter {0}] μP={muP.item():.1e}, "
#                f"μB={muB.mean().item():.1e}, M={M_k:.4e}, ∥r∥={chi_max.mean().item():.4e}")
    
#     for i in range(5):
#         print(f"obj[{i}] = {problem.obj_fn(x).squeeze(-1).squeeze(-1)[i]:.6f}")

#     for i in range(min(5, lb.shape[0])):
#         print(f"Problem {i}:")
#         for j in range(lb.shape[1]):
#             print(f"  Variable {j}: lower = {lb[i, j]}, upper = {ub[i, j]}")
#         print()

#     # NEED TO ADD DIAGNOSTICS FOR X1, X2, Z1, Z2
    
#     # grad_x, grad_s_full, grad_y_full, grad_w_full = problem.merit_grad_M(
#     #         x_col, s_full, y_full, w_full, sE, yE, wE, muP, muB
#     #     )
    
#     # # flatten gradients
#     # grad_x = grad_x.view(B,   n)
#     # grad_s = grad_s_full.view(B, m_tot)
#     # grad_y = grad_y_full.view(B, m_tot)
#     # grad_w = grad_w_full.view(B, m_tot)

#     # inf_x = grad_x.abs().amax(dim=1)   # [B]
#     # inf_s = grad_s.abs().amax(dim=1)   # [B]
#     # inf_y = grad_y.abs().amax(dim=1)   # [B]
#     # inf_w = grad_w.abs().amax(dim=1)   # [B]

#     # print(inf_x.mean().item())
#     # print(inf_s.mean().item())
#     # print(inf_y.mean().item())
#     # print(inf_w.mean().item())

    

#     # outer IPM iterations
#     muB_list = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
#     for muB_outer in muB_list:
#         muB = x.new_tensor([muB_outer])
#         for i in range(1, MAX_OUTER + 1): 
#             # flatten for LSTM call
#             x = x.squeeze(-1)   
#             x1 = x1.squeeze(-1)   
#             x2= x2.squeeze(-1)
#             z1 = z1.squeeze(-1)
#             z2 = z2.squeeze(-1)

#             # currently does 1 micro-iteration of the LSTM, can change that 
#             (x, x1, x2, s, y, v, z1, z2, w), step_loss_vector, (lstm_states, r_k) = models[muB_outer](
#                 problem,
#                 x, x1, x2, s, y, v, z1, z2, w,
#                 x1E.squeeze(-1), x2E.squeeze(-1), sE, yE, vE, z1E.squeeze(-1), z2E.squeeze(-1), wE,
#                 muP, muB, r_k,
#                 lstm_states, bad_x1, bad_x2, bad_z1, bad_z2, M_max, 
#                 project_step=project_tube_all, 
#             )


#             # or just test on 1 problem, and based on its muB, choose the appropriate model
#             # or 


#             # for each problem use diff model based on its value of muB, will have to code that in here. 

#             step_loss = step_loss_vector.sum()

#             print("Loss for this step: ", step_loss)

#             # if train_test == 'train':
#             #     meta_opt.zero_grad(set_to_none=True)      # clear old gradients
#             #     step_loss.backward()                      # back-prop through the micro step
#             #     torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
#             #     meta_opt.step()                           # update the LSTM weights
#             # else:
#             #     val_flag = False

#             # reshape back to columns
#             x = x.unsqueeze(-1)   
#             x1 = x1.unsqueeze(-1)   
#             x2= x2.unsqueeze(-1)
#             z1 = z1.unsqueeze(-1)
#             z2 = z2.unsqueeze(-1)

#             # display merit/residual
#             M_k_new = problem.merit_M(x, x1, x2, s,      
#                     y, v, z1, z2, w,    
#                     x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
#                     muP, muB, bad_x1, bad_x2, bad_z1, bad_z2).mean().item()
            
#             f_val, lb_1, lb_2, lb_3, lb_4, ub_1, ub_2, ub_3, ub_4 = problem.merit_M_indi(x, x1, x2, s,      
#                     y, v, z1, z2, w,    
#                     x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
#                     muP, muB, bad_x1, bad_x2, bad_z1, bad_z2)
#             print("f_val: ", f_val.mean().item())
#             print("lb_1: ", lb_1.mean().item())
#             print("lb_2: ", lb_2.mean().item())
#             print("lb_3: ", lb_3.mean().item())
#             print("lb_4: ", lb_4.mean().item())
#             print("ub_1: ", ub_1.mean().item())
#             print("ub_2: ", ub_2.mean().item())
#             print("ub_3: ", ub_3.mean().item())
#             print("ub_4: ", ub_4.mean().item())


#             total_inner_loss = total_inner_loss + step_loss
            
#             obj_fcn = problem.obj_fn(x).mean().item()

#             print("Objective function value: ", obj_fcn)
            
#             # # diagnostics
#             # s_plus = (s_full + muB).view(-1)                         # [B·m]
#             # w_plus = (w_full + muB).view(-1)                         # [B·m]
#             # d_val  = (c_val + s_full).view(-1)   # c(x)–s,  [B·m]

#             # print(f"[dbg] s+μB min {s_plus.min():.3e} max {s_plus.max():.3e} | "
#             #     f"w+μB min {w_plus.min():.3e} max {w_plus.max():.3e} | "
#             #     f"c-s min {d_val.min():.3e} max {d_val.max():.3e}")

#             chi = problem.chi(x, x1, x2, s,      
#                     y, v, z1, z2, w,    
#                     x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
#                     muP, muB)
#             print(f"[IPM‐iter {muB_outer}] μP={muP.item():.1e}, "
#                 f"μB={muB.mean().item():.1e}, M={M_k_new:.4e}, ∥r∥={chi.mean().item():.4e}")
            
#             print("x: ", x.mean().item())
#             print("z1: ", z1.mean().item())
#             print("z2: ", z2.mean().item())

#             for i in range(5):
#                 print(f"obj[{i}] = {problem.obj_fn(x).squeeze(-1).squeeze(-1)[i]:.6f}")
#             # print("s: ", s_flat.mean().item())
#             # print("y: ", y_flat.mean().item())
#             # print("w: ", w_flat.mean().item())

#             #resid_c = c_val + s_full
#             #y_col   = resid_c / muP
#             #w_col   = muB / (s_col + muB) - y_col

#             # obtaining each of the gradients
#             grad_x, grad_s, grad_y, grad_vv, grad_z1, grad_z2, grad_w = problem.merit_grad_M(
#                 x, x1, x2, s,      
#                     y, v, z1, z2, w,    
#                     x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
#                     muP, muB, bad_x1, bad_x2, bad_z1, bad_z2
#             )
#             # flatten gradients
#             if grad_x is not None: 
#                 grad_x = grad_x.view(B,   n)
#             # if grad_x1 is not None:
#             #     grad_x1 = grad_x1.view(B, n)
#             # if grad_x2 is not None:
#             #     grad_x1 = grad_x1.view(B, n)
#             if grad_z1 is not None:
#                 grad_z1 = grad_z1.view(B, n)
#             if grad_z2 is not None:
#                 grad_z2 = grad_z2.view(B, n)

#             if train_test != "train":
#                 inf_x = grad_x.abs().amax(dim=1)   # [B]
#                 # inf_x1 = grad_x1.abs().amax(dim=1)   # [B]
#                 # inf_x2 = grad_x2.abs().amax(dim=1)   # [B]
#                 inf_z1 = grad_z1.abs().amax(dim=1)   # [B]
#                 inf_z2 = grad_z2.abs().amax(dim=1)   # [B]
#                 if (inf_x < TOL).all() and (inf_z1 < TOL).all() and (inf_z2 < TOL).all(): # stopping criteria
#                     # val += 1
#                     # if (val_stop >= val):
#                     #     val_flag = True
#                     #     return val_flag
#                     # for val flag, need to return something otherwise also
#                     break


#             # primal feasibility
#             P = problem.primal_feasibility(x, x1, x2, s,      
#                     y, v, z1, z2, w,    
#                     x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
#                     muP, muB)      
#             # dual feasibility
#             D = problem.dual_feasibility(x, x1, x2, s,      
#                     y, v, z1, z2, w,    
#                     x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
#                     muP, muB)        
#             # complementarity
#             C = problem.complementarity(x, x1, x2, s,      
#                     y, v, z1, z2, w,    
#                     x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
#                     muP, muB)    
#             print("Primal Feasability: ", P.mean())
#             print("Dual Feasability: ", D.mean()) # stationarity is giving me issues
#             for i in range(5):
#                 print(f"Dual Feasability[{i}] = {D.squeeze(-1).squeeze(-1)[i]:.6f}")
#             for i in range(min(5, lb.shape[0])):
#                 print(f"Problem {i}:")
#                 for j in range(lb.shape[1]):
#                     print(f"  Variable {j}: z1 = {z1[i, j].item()}, z2 = {z2[i, j].item()}")
#                     print(f"obj_grad[{i}]", problem.obj_grad(x)[i])
#                 print()
#             print("Complementarity: ", C.mean())

#             # 1) compute the two scaling norms ‒ D^P and D^B infinity‐norms
#             #    D^P = μP I   →   ||D^P||_∞ = μP
#             #Dp_norm = muP.view(-1)                         # [B] or scalar

#             #    D^B = S_B W_B^{-1},  with S_B = s + μB,  W_B = w + μB
#             #    so each diagonal entry is (s_i+μB)/(w_i+μB) →  ||D^B||_∞ = max_i |(s+μB)/(w+μB)|
#             print(x1.shape)
#             print(z1.shape)
#             print(muB.shape)
#             DB = torch.cat([
#                 ((x1.squeeze(-1) + muB)/(z1.squeeze(-1) + muB)).abs().squeeze(-1),
#                 ((x2.squeeze(-1) + muB)/(z2.squeeze(-1) + muB)).abs().squeeze(-1)
#             ], dim=1)         # [B, 2n]
#             DB_norm = DB.amax(dim=1)  # [B]
#             print(DB_norm.mean())

#             # #print("grade_x shape: ", grad_x_feas.shape)

#             # # 2) check the four M‐iterate conditions (25a)–(25d):
#             # #    ‖∇_x M‖_∞   ≤ τ_k
#             # #    ‖∇_s M‖_∞   ≤ τ_k
#             # #    ‖∇_y M‖_∞   ≤ τ_k · ‖D^P‖_∞
#             # #    ‖∇_w M‖_∞   ≤ τ_k · ‖D^B‖_∞

#             # # first compute the per‐instance infinity norms:
#             # nx = grad_x.abs().amax(dim=1)   # [B]
#             # nz1 = grad_z1.abs().amax(dim=1)   # [B]
#             # nz2 = grad_z2.abs().amax(dim=1)   # [B]


#             # # now form the boolean mask:
#             # cond_M = (
#             #     (nx <= tau_k)        &
#             #     #(nx1 <= tau_k)      &    # primal‐slack test
#             #     #(nx2 <= tau_k)      &    # primal‐slack test
#             #     (nz1 <= tau_k * DB_norm) &
#             #     (nz2 <= tau_k * DB_norm)
#             # )     

#             # print("nx: ", nx.mean().item(), " tau_k: ", tau_k)  
#             # # print("nx1: ", nx1.mean().item(), " tau_k: ", tau_k)    
#             # # print("nx2: ", nx2.mean().item(), " tau_k: ", tau_k)      
#             # print("nz1: ", nz1.mean().item(), " tau_k * DB: ", (tau_k * DB_norm).mean().item()) 
#             # print("nz1: ", nz2.mean().item(), " tau_k * DB: ", (tau_k * DB_norm).mean().item()) 
#             # #print("nz2: ", nz2.mean().item(), " tau_k * Db_norm_2: ", (tau_k * Db_norm_2).mean().item())  

#             Mtest = problem.Mtest(x, x1, x2, s,      
#                     y, v, z1, z2, w,    
#                     x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
#                     muP, muB, M_max, bad_x1, bad_x2, bad_z1, bad_z2)   
            
#             # # print("Variables passing Mtest: ", (Mtest.abs() < 0.1).sum(dim = 0))

#             # # if (Mtest.abs() < 0.1).all():
#             # #     print("SUCCESS")
#             # #     return True 

#             # #if (M_k_new < (0.8*M_k)):
#             # #muB = 0.5*muB

#             # # if (outer % 10 == 0):
#             # #     raw_x1 = (x - lb).clamp(min=eps_t)               # [B,n,1]
#             # #     x1E     = torch.where(mask_lb, raw_x1,
#             # #                         torch.zeros_like(raw_x1))
#             # #     raw_x2 = (ub - x).clamp(min=eps_t)               # [B,n,1]
#             # #     x2E     = torch.where(mask_ub, raw_x2,
#             # #                         torch.zeros_like(raw_x2))
#             # #     z1E = z1.clone()
#             # #     z2E = z2.clone()
#             # #     muB = 0.5*muB

                

#             slack1 = x1.squeeze(-1) + muB        # should stay > 0
#             slack2 = x2.squeeze(-1) + muB        # should stay > 0
            
#             # 2) build a mask of infeasible entries
#             bad_x1 = (slack1 <= 0)   # shape [B,n,1], True wherever x1+μB ≤ 0
#             bad_x2 = (slack2 <= 0)   # shape [B,n,1], True wherever x2+μB ≤ 0

#             total_infeas = int(bad_x1.sum().item())
#             print(f"Total infeasible variables wrt to x1 in batch: {total_infeas}")

#             slackz1 = (z1.squeeze(-1) + muB)    # [B,n]
#             slackz2 = (z2.squeeze(-1) + muB)    # [B,n]

#             # build masks of the exact indices that went non-positive
#             bad_z1 = (slackz1 <= 0).unsqueeze(-1)  # [B,n,1]
#             bad_z2 = (slackz2 <= 0).unsqueeze(-1)  # [B,n,1]

#             # Option A: out-of-place with torch.where
#             z1 = torch.where(bad_z1, torch.zeros_like(z1), z1)
#             z2 = torch.where(bad_z2, torch.zeros_like(z2), z2)
            

#             M_k = M_k_new
                  
    
    
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
    mat_name = "unconstrained_QP"
    file_path = os.path.join("datasets", "qp", f"{mat_name}.mat")
    # make this part a loop so that user can choose

    # train, validation, and test pools
    train_pool = QP(prob_type='QP_unconstrained', learning_type='train', file_path=file_path, seed=args.seed)
    val_pool   = QP(prob_type='QP_unconstrained', learning_type='val', file_path=file_path, seed=args.seed)
    test_pool  = QP(prob_type='QP_unconstrained', learning_type='test', file_path=file_path, seed=args.seed)
    # change the type of function


    # PS_L20_LSTM signature is: __init__(self, n: int, m: int, hidden_size: int=256, num_layers: int=1)
    net = PS_L20_LSTM(
        problem = train_pool, 
        #m_ineq=DIM_S,
        #m_eq=DIM_S,
        hidden_size=HIDDEN,
        #num_layers=K_INNER,   
        K_inner = K_INNER,  
        device=DEVICE
    ).to(DEVICE)
    # training loop
    if args.mode == "train":
        meta_opt = optim.Adam(net.parameters(), lr=LR_META) # adam optimizer
        import copy
        import numpy as np

        BATCH = 200
        N     = train_pool.Q.shape[0]
        for ep in range(1, EPOCHS+1):
            print("EPOCH: ", ep)
            t0 = time.time()
            perm = torch.arange(N, device=train_pool.Q.device)
            for i in range(0, N, BATCH):
                print("BATCH: ", N/BATCH)
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
                solve_one_qcqp(meta_opt, mini, net)
                # get the objective 
                # evaluate(val_pool, net, tag="val")
            print(f"Epoch {ep:3d} | {time.time()-t0:.1f}s")
            solve_one_qcqp(meta_opt, val_pool, net, train_test = 'val')
        torch.save(net.state_dict(), args.ckpt)
        print(f"✓ weights saved to {args.ckpt}")

    # test evaluation 
    else:
        net.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
        print(f"✓ loaded weights from {args.ckpt}")
        net.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
        net.to(DEVICE)
        net.eval()

        # 3) run each test problem through solve_one_qcqp (no grad!)
        total_obj = 0.0
        with torch.no_grad():
            meta_opt = optim.Adam(net.parameters(), lr=LR_META) # adam optimizer
            # solve_one_qcqp should return the final x (and maybe status/info)
            solve_one_qcqp(meta_opt, test_pool, net, train_test = 'test')
            # obj_val = test_pool.obj_fn(x_star).item()
            # total_obj += obj_val

        print(f"\nAverage test objective: {total_obj/len(test_pool):.6f}")

    #evaluate(test_pool, net, tag="test")

