import argparse, time, torch, torch.optim as optim, copy, random, re
import torch.nn.init as init
import numpy as np
import json

import os
from problems.Convex_QCQP import Convex_QCQP        
from problems.QP_extended import QP
from models.LSTM_L20_Projected_Math_extended import PS_L20_LSTM         

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
# push all ad-hoc tensor creations to the chosen device by default
try:
    torch.set_default_device(DEVICE)
except AttributeError:
    pass  # older torch; fall back to per-call device args

#DIM_X    = 10        # decision variables
DIM_S    = 50         # inequality slacks
HIDDEN   = 128        # LSTM hidden size
K_INNER  = 1        

EPOCHS   = 200                    # meta-training epochs
LR_META  = 1e-4                 # learning-rate for Adam

MAX_OUTER  = 50
tolP, tolD, tolC = 1e-4, 1e-4, 1e-4
TOL = 1e-4
S1_MAX_E = 1e5
W1_MAX_E = 1e5
Y_MAX_E = 1e5  
V_MAX_E = 1e5

TOLM = 0.1

Z1_MAX_E = 1e5
Z2_MAX_E = 1e5   
X1_MAX_E = 1e5
X2_MAX_E = 1e5  
X_MAX_E = 1e5

def project_tube_all(
        problem,
        xk, sk, yk, vk, z1k, z2k, w1k, w2k,            
        x,  s,  y,  vv, z1,  z2,  w1, w2,            
        x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E,    
        muP, muB, muA, 
        sigma: float = 0.80, 
        eps = 1e-6): 

    def proj_lower(v, v_k, ell):
        wall = torch.minimum(v_k - sigma * (v_k - ell) , torch.full_like(v_k, 0))
        lower = wall 
        return torch.maximum(v, lower)

    B, device = x.shape[0], x.device
    n         = x.shape[1]

    xk  = xk.squeeze(-1);   x  = x.squeeze(-1)
    z1k = z1k.squeeze(-1); z1 = z1.squeeze(-1)
    z2k = z2k.squeeze(-1); z2 = z2.squeeze(-1)

    lb = problem.lb
    ub = problem.ub
    if lb.dim() == 1:
        lb = lb.unsqueeze(0).expand(B, n)
    if ub.dim() == 1:
        ub = ub.unsqueeze(0).expand(B, n)

    mask_lb = torch.isfinite(lb)          
    mask_ub = torch.isfinite(ub)         

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

    x_p = x_flat.unsqueeze(-1).clamp_min(-X_MAX_E).clamp_max(X_MAX_E)

    #s_p = proj_lower_negwall(s.squeeze(-1),  sk.squeeze(-1), ell_sw).unsqueeze(-1)
    #w_p = proj_lower_negwall(w.squeeze(-1),  wk.squeeze(-1), ell_sw).unsqueeze(-1)

    z1_p_raw = torch.where(mask_lb,
                        proj_lower(z1,  z1k, -muB),
                        z1)                # shape [B,n]
    z1_p = torch.clamp(z1_p_raw, max=Z1_MAX_E)  # clamp above at 1e10
    z1_p = z1_p.unsqueeze(-1)               # shape [B,n,1]
    # two‐step version for clarity
    z2_p_raw = torch.where(mask_ub,
                        proj_lower(z2, z2k, -muB),
                        z2)                # shape [B,n]
    z2_p = torch.clamp(z2_p_raw, max=Z2_MAX_E)  # clamp above at 1e10
    z2_p = z2_p.unsqueeze(-1)               # shape [B,n,1]


    # project s to lower wall -μB  (c(x) ≥ 0 ⇒ s has implicit lower bound)
    if problem.num_ineq > 0:
        sk  = sk.squeeze(-1)  if sk.dim()  == 3 else sk
        s   = s.squeeze(-1)   if s.dim()   == 3 else s
        yk  = yk.squeeze(-1)  if yk.dim()  == 3 else yk
        y   = y.squeeze(-1)   if y.dim()   == 3 else y
        #vk  = vk.squeeze(-1)  if vk.dim()  == 3 else vk
        #vv  = vv.squeeze(-1)  if vv.dim()  == 3 else vv
        w1k = w1k.squeeze(-1) if (w1k is not None and w1k.dim()==3) else w1k
        w1  = w1.squeeze(-1)  if (w1  is not None and w1.dim()==3)  else w1
        s_p_raw = proj_lower(s,  sk,  -muB)         # shape [B, m_ineq]
        #print("s_p_raw: ", s_p_raw)
        s_p     = torch.clamp(s_p_raw, max=S1_MAX_E).unsqueeze(-1)
        # y, vv: no tube walls (ℓ = -∞), pass through
        y_p = torch.clamp(y, -Y_MAX_E, max = Y_MAX_E)
        y_p  = y_p.unsqueeze(-1)  if y_p.dim()==2  else y_p
        #print("s_p: ", s_p_raw)
    else:
        s_p = None
        y_p = None
    
    if problem.num_eq > 0:
        v_p = torch.clamp(vv, -V_MAX_E, max = V_MAX_E)
        vv_p = v_p.unsqueeze(-1) if v_p.dim()==2 else v_p
    else:
        vv_p = None

    # project w1 to lower wall -μB  (implicit bound on w)
    if (problem.num_ineq > 0) and (w1 is not None):
        w1_p_raw = proj_lower(w1, w1k, -muB)        # shape [B, m_ineq]
        w1_p     = torch.clamp(w1_p_raw, max=W1_MAX_E).unsqueeze(-1)
    else:
        w1_p = None

    return x_p, s_p, y_p, vv_p, z1_p, z2_p, w1_p

def solve_one_qcqp(meta_opt, problem: "QP",
                   net: PS_L20_LSTM,
                   muP0 = 1, muB0 = 1, muA0 = 1, 
                   max_outer=MAX_OUTER, train_test = 'train', val_stop = 5, problem_type = 'normal', print_level = 3,
                   rollout=True, return_info: bool = False, backprop_every: int = 1):

    # get the print level in so that you can actually compare the time
    # fix the error tomorrow
    # run IPM-LSTM results in 

    if train_test == 'train':
        print("train")
    else: 
        print("test")
    B = problem.Q.shape[0]
    n_vec = problem.n_vec
    n, m_ineq, m_eq = problem.num_var, problem.num_ineq, problem.num_eq
    device = DEVICE
    has_cons = (m_ineq > 0) or (m_eq > 0)

    eps_t = torch.tensor(1e-10, device=device)

    lb_vals = problem.lb.expand(B, n).to(device)   
    ub_vals = problem.ub.expand(B, n).to(device)   

    lb = lb_vals.unsqueeze(-1)
    ub = ub_vals.unsqueeze(-1)

    mask_lb = torch.isfinite(lb_vals)  
    mask_ub = torch.isfinite(ub_vals)  

    # create branch for type of initialization based on the type of problem and its purpose
    if problem_type == 'normal':
        if problem.num_lb != 0 or problem.num_ub != 0:
            # initialize x
            lb_vals = problem.lb.expand(B, n).to(device)    # [B,n]
            ub_vals = problem.ub.expand(B, n).to(device)    # [B,n]

            # masks for which bounds are finite
            has_lb = torch.isfinite(lb_vals).to(device)                    # [B,n]
            has_ub = torch.isfinite(ub_vals).to(device)                    # [B,n]

            # start with zeros
            x_init = torch.zeros((B, n), device=device)       # [B,n]

            # per-coordinate initialization based on which bounds are present
            both    = has_lb & has_ub           # midpoint when both finite
            lb_only = has_lb & ~has_ub          # nudge above lower
            ub_only = ~has_lb & has_ub          # nudge below upper

            x_init[both]    = 0.5 * (lb_vals[both] + ub_vals[both])
            x_init[lb_only] = lb_vals[lb_only] + 1.0
            x_init[ub_only] = ub_vals[ub_only] - 1.0

            x = x_init.unsqueeze(-1)

            xE = x.clone()
            # if problem.num_eq > 0:
            #     A_pinv = torch.linalg.pinv(problem.A) 
            #     x  = torch.bmm(A_pinv, problem.b).to(device)
            # else:
            #     x = x_init.unsqueeze(-1)
            # xE = x.clone()

            # initialize the duals anbd the shifts            
            muB = x.new_tensor([muB0], device=device)
            muP = x.new_tensor([muP0], device=device)
            #tau_k = 2
            tolM = torch.tensor(TOLM).to(device)

            # initialize x1, x2, z1, and z2
            # masks for where bounds are finite (broadcast to [B,n,1])
            mask_lb = mask_lb.unsqueeze(-1)   # [B,n,1], True where ℓ_j > -∞
            mask_ub = mask_ub.unsqueeze(-1)   # [B,n,1], True where u_j < +∞

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
        
        # initializing the variables, will need initialize v also here later 
        if m_eq > 0:
            A_pinv = torch.linalg.pinv(problem.A)       # [B,n,p]
            #x  = torch.bmm(A_pinv, problem.b).to(device)        # [B,n,1]
            # A = problem.A.to(device)                            # [B, meq, n]
            # b = problem.b.to(device); b = b.unsqueeze(-1) if b.dim()==2 else b  # [B,meq,1]
            # At  = A.transpose(1, 2)                             # [B, n, meq]
            # AAt = torch.bmm(A, At)                              # [B, meq, meq]
            # Ieq = torch.eye(AAt.size(1), device=device).unsqueeze(0).expand_as(AAt)
            # rhs = torch.bmm(A, x) - b                           # [B, meq, 1]
            # y_eq = torch.linalg.solve(AAt + 1e-2*Ieq, rhs)    # [B, meq, 1]
            # x    = x - torch.bmm(At, y_eq)                      # [B, n, 1]
            muP = x.new_tensor([muP0], device=device)   
            v  = torch.zeros(B, m_eq, 1, device=device)   # start centered
            vE = v.clone()
            v_max_e = torch.tensor(V_MAX_E, device=device)

        else:
            #x  = torch.zeros(B, n, 1, device=device)
            v = None
            vE = None
            # still keep primal barrier values when no equalities
            muP = x.new_tensor([muP0], device=device)
        muA = x.new_tensor([muA0], device=device)   
        
        

        # slack variable s initialization for the inequalities
        if m_ineq > 0:
            g0    = problem.ineq_resid(x).to(device)           # [B,m_ineq,1]
            s = torch.clamp(g0, min=1e-6)          # [B,m_ineq,1] 
            #print(g0)
            #s = -g0
            muP = x.new_tensor([muP0], device=device)   

            # Use μ^P for (y, s) complementarity: diag(y) diag(s) e = μ^P e  ⇒  y = μ^P / s
            y  = (muP / s.clamp(min=eps_t))
            yE = y.clone()

            # ----- bounds for s and their (slack,dual): (s1,w1) for lower, (s2,w2) for upper -----
            # Defaults: s ≥ 0 (lower bound 0), no upper bound unless provided on problem
            # s_lb = getattr(problem, 's_lb', torch.zeros(B, m_ineq, 1, device=device))
            # s_ub = getattr(problem, 's_ub', torch.full((B, m_ineq, 1), float('inf'), device=device))

            # mask_s_lb = torch.isfinite(s_lb)
            # mask_s_ub = torch.isfinite(s_ub)

            # # lower-bound slack/dual for s
            # raw_s1 = (s - s_lb).clamp(min=eps_t)
            # s1     = torch.where(mask_s_lb, raw_s1, torch.zeros_like(raw_s1))
            # w1     = torch.where(mask_s_lb, (muB / s1.clamp(min=eps_t)), torch.zeros_like(s1))

            # # upper-bound slack/dual for s (often none)
            # raw_s2 = (s_ub - s).clamp(min=eps_t)
            # s2     = torch.where(mask_s_ub, raw_s2, torch.zeros_like(raw_s2))
            # w2     = torch.where(mask_s_ub, (muB / s2.clamp(min=eps_t)), torch.zeros_like(s2))
            s1 = s.clone()
            w1     = muB / s1.clamp(min=eps_t)

            s2 = None
            w2 = None

            # shifted copies (E-variables) — start at current values
            s1E, s2E = s1.clone(), None
            w1E, w2E = w1.clone(), None
            s1_max_e = torch.tensor(S1_MAX_E, device=device)
            w1_max_e = torch.tensor(W1_MAX_E, device=device)
            y_max_e = torch.tensor(Y_MAX_E, device=device)
        else:

            s = None
            y = None
            w1 = None
            w2 = None
            yE = None
            s1E = None
            s2E = None
            w1E = None
            w2E = None
    
    elif problem_type == 'portfolio':
        B = problem.Q.shape[0]
        n, m_ineq, m_eq = problem.num_var, 0, problem.num_eq
        device = DEVICE
        eps = 1e-10
        eps_t = torch.tensor(eps, device=device)

        Q = problem.Q.to(device)                          # [B,n,n]
        p = problem.p.to(device)
        if p.dim() == 2: p = p.unsqueeze(-1)              # [B,n,1]

        # bounds (x has lb>=0 on first s coords; y is free)
        lb = problem.lb.to(device)
        if lb.dim() == 2: lb = lb.unsqueeze(-1)           # [B,n,1]
        ub = torch.full_like(lb, float('inf'))            # [B,n,1]
        has_lb = torch.isfinite(lb)
        has_ub = torch.isfinite(ub)

        # equalities
        if m_eq > 0:
            A = problem.A.to(device)                      # [B,meq,n]
            b = problem.b.to(device)                      # [B,meq,1]
            At = A.transpose(1, 2)                        # [B,n,meq]
        else:
            A = b = At = None

        # ---- sizes: n = s + t, and m_eq = t + 1 (rows: y - F x = 0, and 1^T x = 1)
        assert m_eq >= 1, "portfolio problems expect at least the sum-to-one equality"
        t = m_eq - 1
        s = n - t
        assert s > 0 and t >= 0, "inconsistent (n, m_eq) for portfolio structure"

        # ---- build x so that:
        #  (i) x_assets >= 0 and sum(x_assets) = 1
        #  (ii) y = F x_assets with F deduced from A: rows 0..t-1 encode y - F x = 0 → A[:t,:s] = -F, A[:t,s:] = I
        # sample positive interior point on simplex
        x_assets = torch.rand(B, s, 1, device=device) + 1e-3
        x_assets = x_assets / (x_assets.sum(dim=1, keepdim=True) + 1e-12)  # sum to 1

        # respect any finite lower bounds on x (first s). If lb>0 exists, nudge and renormalize.
        lb_assets = lb[:, :s, :]                                           # [B,s,1]
        mask_lb_assets = torch.isfinite(lb_assets)
        if mask_lb_assets.any():
            # make sure x_assets ≥ lb_assets (feasible if sum(lb_assets) ≤ 1)
            slack = (1.0 - torch.clamp(lb_assets, min=0.0)).sum(dim=1, keepdim=True)  # [B,1,1]
            # redistribute remaining mass proportionally to current positive vector
            base = torch.clamp(x_assets, min=1e-6)
            base = base / (base.sum(dim=1, keepdim=True) + 1e-12)
            x_assets = torch.clamp(lb_assets, min=0.0) + torch.clamp(slack, min=0.0) * base
            # tiny safety renorm
            x_assets = x_assets / (x_assets.sum(dim=1, keepdim=True) + 1e-12)

        # compute F from A (first t rows): A[:t,:s] = -F, A[:t,s:] = I_t
        if t > 0:
            F = -A[:, :t, :s]                                             # [B,t,s]
            y_factors = torch.bmm(F, x_assets)                            # [B,t,1]
            x = torch.cat([x_assets, y_factors], dim=1)                   # [B,n,1]
        else:
            x = x_assets                                                  # no factor block

        # optional small interior push for box on x (only lb on first s entries)
        x[:, :s, :] = torch.maximum(x[:, :s, :], lb[:, :s, :] + 1e-6)

        xE = x.clone()

        # barriers and tolerances (unchanged)
        muB = x.new_tensor([muB0], device=device)         # [1]
        muP = x.new_tensor([muP0], device=device)         # [1]
        muA = x.new_tensor([muA0], device=device)         # [1]
        tolM = torch.tensor(TOLM, device=device)

        # bound slacks/duals for box (y is free so only first s may be active)
        raw_x1 = (x - lb).clamp_min(eps)
        x1 = torch.where(has_lb, raw_x1, torch.zeros_like(raw_x1))
        z1 = torch.where(has_lb, muB / x1, torch.zeros_like(x1))

        raw_x2 = (ub - x).clamp_min(eps)
        x2 = torch.where(has_ub, raw_x2, torch.zeros_like(raw_x2))
        z2 = torch.where(has_ub, muB / x2, torch.zeros_like(x2))

        x1E, x2E, z1E, z2E = x1.clone(), x2.clone(), z1.clone(), z2.clone()
        x1_max_e = torch.tensor(X1_MAX_E, device=device)
        x2_max_e = torch.tensor(X2_MAX_E, device=device)
        z1_max_e = torch.tensor(Z1_MAX_E, device=device)
        z2_max_e = torch.tensor(Z2_MAX_E, device=device)

        # equality multipliers (leave as in your code)
        if m_eq > 0:
            g = torch.bmm(Q, x) + p                                       # [B,n,1]
            At = A.transpose(1, 2)                                        # [B,n,meq]
            At_pinv = torch.linalg.pinv(At)                               # [B,meq,n]
            v = -torch.bmm(At_pinv, g)                                    # [B,meq,1]
            vE = v.clone()
            v_max_e = torch.tensor(V_MAX_E, device=device)
        else:
            v = vE = None

        # no inequalities in this branch
        s = y = w1 = w2 = None
        s1E = s2E = yE = w1E = w2E = None






    muB = muB.expand(B, 1)
    muP = muP.expand(B, 1)
    muA = muA.expand(B, 1)
    tolM = tolM.expand(B)
    
    


    L = net.lstm_px.num_layers
    H = net.lstm_px.hidden_size

    BN = B * n
    h_x = torch.zeros(L, BN, H, device=device)
    c_x = torch.zeros(L, BN, H, device=device)

    # Xavier‐uniform initialization for the hidden and cell state of LSTM
    init.xavier_uniform_(h_x)
    init.xavier_uniform_(c_x)
    lstm_states_x = (h_x, c_x)

    L = net.lstm_ps.num_layers
    H = net.lstm_ps.hidden_size

    lstm_states_s = None
    if (m_ineq > 0):
        Bm = B * s.shape[1]
        h_s = torch.zeros(L, Bm, H, device=device)
        c_s = torch.zeros(L, Bm, H, device=device)
        init.xavier_uniform_(h_s)
        init.xavier_uniform_(c_s)
        lstm_states_s = (h_s, c_s)

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
    _add_block(w1,  'w1')    # ineq‐dual
    _add_block(w2,  'w2')    # ineq‐dual
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

    # total_infeas = int(bad_x1.sum().item())
    # #print(f"Total infeasible variables wrt to x1 in batch: {total_infeas}")

    # slackz1 = (z1.squeeze(-1) + muB)    # [B,n]
    # slackz2 = (z2.squeeze(-1) + muB)    # [B,n]

    # # build masks of the exact indices that went non-positive
    # bad_z1 = (slackz1 <= 0).unsqueeze(-1)  # [B,n,1]
    # bad_z2 = (slackz2 <= 0).unsqueeze(-1)  # [B,n,1]

    # # Option A: out-of-place with torch.where
    # z1 = torch.where(bad_z1, torch.zeros_like(z1), z1)
    # z2 = torch.where(bad_z2, torch.zeros_like(z2), z2)             

    total_inner_loss = 0.0
    unroll_losses = []
    conv_time = None
    start_time_total = None
    last_ok = None
    obj_traj_t = []
    obj_traj_v = []
    if return_info:
        conv_time = torch.full((B,), float("inf"), device=device)
        start_time_total = time.perf_counter()
        last_ok = torch.zeros(B, dtype=torch.bool, device=device)
    BACKPROP_EVERY = max(1, int(backprop_every))
    detach_next = random.randint(1, BACKPROP_EVERY)

    
    M_max = problem.merit_M(x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA, bad_x1, bad_x2)
    
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


    chi = problem.chi(x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA)
    chi_max = chi.clone()
    initial_chi = chi.clone()
    initial_state = None
    best_state = None
    best_chi = None

    def _clone_detach(t):
        return t.detach().clone() if t is not None else None

    def _mask_view(mask, target_dim):
        # reshape a [B]-mask so it broadcasts over tensors with `target_dim` dims
        return mask.view([-1] + [1] * (target_dim - 1))

    def _align_to(curr_t, template_t):
        # match `template_t`'s shape to `curr_t` when they only differ by a trailing singleton
        if curr_t is None or template_t is None:
            return None
        if template_t.shape == curr_t.shape:
            return template_t
        if (template_t.dim() == curr_t.dim() + 1
                and template_t.shape[-1] == 1
                and template_t.shape[:-1] == curr_t.shape):
            return template_t.squeeze(-1)
        if (template_t.dim() + 1 == curr_t.dim()
                and curr_t.shape[-1] == 1
                and curr_t.shape[:-1] == template_t.shape):
            return template_t.unsqueeze(-1)
        return None

    def _restart_from_init(mask, init_state, noise_scale=0.0001, lb=None, ub=None):
        """
        Restart a batch of regressed samples from their initial state with optional noise.
        Returns updated (x, x1, x2, z1, z2, s, s1, w1, y, v, muB, muP, x1E, x2E, s1E, yE, vE, z1E, z2E, w1E).
        """
        def _perturb(t, span=None):
            if t is None:
                return None
            if noise_scale and noise_scale > 0:
                if span is None:
                    span = torch.ones_like(t, device=t.device)
                jitter = torch.randn_like(t) * noise_scale * span.clamp_min(1e-3)
                return t + jitter
            return t

        # primal x
        x_init = _align_to(x, init_state["x"])
        x_new = x_init
        if x_new is not None:
            span = None
            if lb is not None and ub is not None:
                span = ub - lb
                span = torch.where(torch.isfinite(span), span, torch.ones_like(span, device=x_new.device))
                if x_new.dim() > span.dim():
                    span = span.unsqueeze(-1)
            x_new = _perturb(x_new, span)
            if lb is not None:
                lb_aligned = lb if lb.dim() == x_new.dim() else lb.unsqueeze(-1)
                x_new = torch.where(torch.isfinite(lb_aligned), torch.maximum(x_new, lb_aligned), x_new)
            if ub is not None:
                ub_aligned = ub if ub.dim() == x_new.dim() else ub.unsqueeze(-1)
                x_new = torch.where(torch.isfinite(ub_aligned), torch.minimum(x_new, ub_aligned), x_new)

        # recompute bound slacks/duals from the (possibly perturbed) x
        lb_vals_local = lb if lb is not None else None
        ub_vals_local = ub if ub is not None else None
        if lb_vals_local is not None and lb_vals_local.dim() == 2:
            lb_vals_local = lb_vals_local.unsqueeze(-1)
        if ub_vals_local is not None and ub_vals_local.dim() == 2:
            ub_vals_local = ub_vals_local.unsqueeze(-1)

        mask_lb_local = torch.isfinite(lb_vals_local) if lb_vals_local is not None else None
        mask_ub_local = torch.isfinite(ub_vals_local) if ub_vals_local is not None else None

        x1_new = None
        z1_new = None
        if lb_vals_local is not None:
            raw_x1 = (x_new - lb_vals_local).clamp(min=eps_t)
            x1_new = torch.where(mask_lb_local, raw_x1, torch.zeros_like(raw_x1))
        x2_new = None
        z2_new = None
        if ub_vals_local is not None:
            raw_x2 = (ub_vals_local - x_new).clamp(min=eps_t)
            x2_new = torch.where(mask_ub_local, raw_x2, torch.zeros_like(raw_x2))

        muB_new = torch.ones_like(muB) if muB is not None else None
        muP_new = torch.ones_like(muP) if muP is not None else None

        if x1_new is not None:
            z1_new = torch.where(mask_lb_local, muB_new.unsqueeze(-1) / x1_new.clamp(min=eps_t), torch.zeros_like(x1_new))
        if x2_new is not None:
            z2_new = torch.where(mask_ub_local, muB_new.unsqueeze(-1) / x2_new.clamp(min=eps_t), torch.zeros_like(x2_new))

        # inequality slack/dual recomputed if available
        s_new = None
        y_new = None
        s1_new = None
        w1_new = None
        if init_state.get("s") is not None:
            x_for_resid = x_new.unsqueeze(-1) if x_new.dim() == 2 else x_new
            g_restart = problem.ineq_resid(x_for_resid)
            g_restart = g_restart.squeeze(-1) if g_restart.dim() == 3 else g_restart
            s_new = torch.clamp(g_restart, min=1e-6)
            s1_new = s_new.clone()
            y_new = muP_new.unsqueeze(-1) / s_new.clamp(min=eps_t)
            w1_new = muB_new.unsqueeze(-1) / s1_new.clamp(min=eps_t)

        v_new = None
        if init_state.get("v") is not None:
            v_new = init_state["v"].detach().clone()

        # apply mask to select which batch items restart, aligning shapes to current tensors
        def _align_for_where(curr, new):
            if new is None:
                return None
            if curr is None:
                return new
            aligned = _align_to(curr, new)
            return aligned if aligned is not None else new

        x_new = _align_for_where(x, x_new)
        mv = _mask_view(mask, x_new.dim())
        x_new = torch.where(mv, x_new, x)
        x1_new = _align_for_where(x1, x1_new)
        if x1_new is not None:
            mv_x1 = _mask_view(mask, x1_new.dim())
            x1_new = torch.where(mv_x1, x1_new, x1) if x1 is not None else x1_new
        x2_new = _align_for_where(x2, x2_new)
        if x2_new is not None:
            mv_x2 = _mask_view(mask, x2_new.dim())
            x2_new = torch.where(mv_x2, x2_new, x2) if x2 is not None else x2_new
        z1_new = _align_for_where(z1, z1_new)
        if z1_new is not None:
            mv_z1 = _mask_view(mask, z1_new.dim())
            z1_new = torch.where(mv_z1, z1_new, z1) if z1 is not None else z1_new
        z2_new = _align_for_where(z2, z2_new)
        if z2_new is not None:
            mv_z2 = _mask_view(mask, z2_new.dim())
            z2_new = torch.where(mv_z2, z2_new, z2) if z2 is not None else z2_new
        s_new = _align_for_where(s, s_new)
        if s_new is not None:
            mv_s = _mask_view(mask, s_new.dim())
            s_new = torch.where(mv_s, s_new, s) if s is not None else s_new
        s1_new = _align_for_where(s1E, s1_new)
        if s1_new is not None:
            mv_s1 = _mask_view(mask, s1_new.dim())
            s1_new = torch.where(mv_s1, s1_new, s1E) if s1E is not None else s1_new
        w1_new = _align_for_where(w1, w1_new)
        if w1_new is not None:
            mv_w1 = _mask_view(mask, w1_new.dim())
            w1_new = torch.where(mv_w1, w1_new, w1) if w1 is not None else w1_new
        y_new = _align_for_where(y, y_new)
        if y_new is not None:
            mv_y = _mask_view(mask, y_new.dim())
            y_new = torch.where(mv_y, y_new, y) if y is not None else y_new
        v_new = _align_for_where(v, v_new)
        if v_new is not None:
            mv_v = _mask_view(mask, v_new.dim())
            v_new = torch.where(mv_v, v_new, v) if v is not None else v_new

        muB_new = torch.where(mask.unsqueeze(-1), muB_new, muB)
        muP_new = torch.where(mask.unsqueeze(-1), muP_new, muP)

        # align outputs to reference shapes so downstream code sees the same dims
        def _align_out(ref, new):
            if new is None:
                return None
            aligned = _align_to(ref, new)
            return aligned if aligned is not None else new

        x_new  = _align_out(x, x_new)
        x1_new = _align_out(x1, x1_new)
        x2_new = _align_out(x2, x2_new)
        z1_new = _align_out(z1, z1_new)
        z2_new = _align_out(z2, z2_new)
        s_new  = _align_out(s, s_new)
        s1_new = _align_out(s1E, s1_new) if 's1E' in locals() else s1_new
        w1_new = _align_out(w1, w1_new)
        y_new  = _align_out(y, y_new)
        v_new  = _align_out(v, v_new)
        muB_new = _align_out(muB, muB_new)
        muP_new = _align_out(muP, muP_new)

        # estimates cloned from current variables, aligned to their refs
        x1E_new = _align_out(x1E, x1_new.clone() if x1_new is not None else None)
        x2E_new = _align_out(x2E, x2_new.clone() if x2_new is not None else None)
        s1E_new = _align_out(s1E, s1_new.clone() if s1_new is not None else None)
        yE_new  = _align_out(yE,  y_new.clone()  if y_new  is not None else None)
        vE_new  = _align_out(vE,  v_new.clone()  if v_new  is not None else None)
        z1E_new = _align_out(z1E, z1_new.clone() if z1_new is not None else None)
        z2E_new = _align_out(z2E, z2_new.clone() if z2_new is not None else None)
        w1E_new = _align_out(w1E, w1_new.clone() if w1_new is not None else None)

        return (x_new, x1_new, x2_new, z1_new, z2_new,
                s_new, s1_new, w1_new, y_new, v_new,
                muB_new, muP_new,
                x1E_new, x2E_new, s1E_new, yE_new, vE_new, z1E_new, z2E_new, w1E_new)

    def _update_best_tensor(best_t, curr_t, mask):
        if best_t is None or curr_t is None:
            return best_t
        curr_detached = _clone_detach(curr_t)
        mask_view = mask.view([-1] + [1] * (curr_detached.dim() - 1))
        return torch.where(mask_view, curr_detached, best_t)

    def _snapshot_state():
        return {
            "x": _clone_detach(x),
            "s": _clone_detach(s),
            "y": _clone_detach(y),
            "v": _clone_detach(v),
            "z1": _clone_detach(z1),
            "z2": _clone_detach(z2),
            "w1": _clone_detach(w1),
            "w2": _clone_detach(w2),
            "x1E": _clone_detach(x1E),
            "x2E": _clone_detach(x2E),
            "s1E": _clone_detach(s1E),
            "s2E": _clone_detach(s2E),
            "yE": _clone_detach(yE),
            "vE": _clone_detach(vE),
            "z1E": _clone_detach(z1E),
            "z2E": _clone_detach(z2E),
            "w1E": _clone_detach(w1E),
            "w2E": _clone_detach(w2E),
            "muP": _clone_detach(muP),
            "muB": _clone_detach(muB),
            "muA": _clone_detach(muA),
        }

    def _revert_tensor(curr_t, prev_t, mask):
        if curr_t is None or prev_t is None:
            return curr_t
        if curr_t.shape != prev_t.shape:
            # allow trailing singleton differences (e.g., [B, n] vs [B, n, 1])
            if (prev_t.dim() + 1 == curr_t.dim()) and (curr_t.shape[-1] == 1) and (prev_t.shape == curr_t.shape[:-1]):
                prev_t = prev_t.unsqueeze(-1)
            elif (prev_t.dim() == curr_t.dim() + 1) and (prev_t.shape[-1] == 1) and (prev_t.shape[:-1] == curr_t.shape):
                prev_t = prev_t.squeeze(-1)
            else:
                return curr_t  # shape mismatch; skip revert for this block
        mask_view = mask.view([-1] + [1] * (curr_t.dim() - 1))
        return torch.where(mask_view, prev_t, curr_t)

    def _print_best_state_metrics():
        if train_test == "train" or best_state is None:
            return
        B_local = best_state["x"].shape[0]
        best_P = problem.primal_feasibility(
            best_state["x"], best_state["s"],
            best_state["y"], best_state["v"], best_state["z1"], best_state["z2"], best_state["w1"], best_state["w2"],
            best_state["x1E"], best_state["x2E"], best_state["s1E"], best_state["s2E"], best_state["yE"], best_state["vE"], best_state["z1E"], best_state["z2E"], best_state["w1E"], best_state["w2E"],
            best_state["muP"], best_state["muB"], best_state["muA"],
        )
        best_D = problem.dual_feasibility(
            best_state["x"], best_state["s"],
            best_state["y"], best_state["v"], best_state["z1"], best_state["z2"], best_state["w1"], best_state["w2"],
            best_state["x1E"], best_state["x2E"], best_state["s1E"], best_state["s2E"], best_state["yE"], best_state["vE"], best_state["z1E"], best_state["z2E"], best_state["w1E"], best_state["w2E"],
            best_state["muP"], best_state["muB"], best_state["muA"],
        )
        best_C = problem.complementarity(
            best_state["x"], best_state["s"],
            best_state["y"], best_state["v"], best_state["z1"], best_state["z2"], best_state["w1"], best_state["w2"],
            best_state["x1E"], best_state["x2E"], best_state["s1E"], best_state["s2E"], best_state["yE"], best_state["vE"], best_state["z1E"], best_state["z2E"], best_state["w1E"], best_state["w2E"],
            best_state["muP"], best_state["muB"], best_state["muA"],
        )
        x_for_obj = best_state["x"].unsqueeze(-1) if best_state["x"].dim() == 2 else best_state["x"]
        best_obj_mean = problem.obj_fn(x_for_obj).mean().item()
        print(f"Best-chi state means -> P: {best_P.mean().item():.4e}, D: {best_D.mean().item():.4e}, C: {best_C.mean().item():.4e}")
        print(f"Best-state objective mean: {best_obj_mean:.6f}")
        thresh = 1e-2
        ok = (best_P < thresh) & (best_D < thresh) & (best_C < thresh)
        print("Best-chi Percentage converged: ", (ok.sum() / B_local))
        for i in range(min(10, B_local)):
            print(f"best[{i}] -> P: {best_P.view(-1)[i].item():.4e}, D: {best_D.view(-1)[i].item():.4e}, C: {best_C.view(-1)[i].item():.4e}")

        def _print_infeas_stats(label, vals):
            vals_flat = vals.view(-1)
            mean_v = vals_flat.mean().item()
            max_v = vals_flat.max().item()
            median_v = vals_flat.median().item()
            pct95_v = torch.quantile(vals_flat, 0.95).item()
            print(f"{label} stats -> mean: {mean_v:.4e}, max: {max_v:.4e}, median: {median_v:.4e}, p95: {pct95_v:.4e}")

        # bound constraint violation (L2 over slacks) for the best state
        x_best_plain = best_state["x"].squeeze(-1) if best_state["x"].dim() == 3 else best_state["x"]
        x1_best = problem.lower_bound_dist(x_best_plain)  # [B,n]
        x2_best = problem.upper_bound_dist(x_best_plain)  # [B,n]
        # per-coordinate max(x1, x2), then L2 per problem
        bc_coord = torch.maximum(x1_best, x2_best)        # [B,n]
        bc_violation = bc_coord.flatten(1).norm(p=2, dim=1)  # [B]
        print(f"Bound constraint violation -> mean: {bc_violation.mean().item():.4e}, max: {bc_violation.max().item():.4e}")
        # bound violations for converged subset
        if ok.any():
            bc_ok = bc_violation[ok]
            print(f"Bound violation (converged) -> mean: {bc_ok.mean().item():.4e}, max: {bc_ok.max().item():.4e}")
            # objective on converged subset only (match constraint stats)
            x_for_obj = best_state["x"].unsqueeze(-1) if best_state["x"].dim() == 2 else best_state["x"]
            # slice Q and p to align batch dimension with converged subset to avoid shape mismatch
            Q_ok = problem.Q[ok] if hasattr(problem, "Q") else None
            p_ok = problem.p[ok] if hasattr(problem, "p") else None
            obj_ok_mean = problem.obj_fn(x_for_obj[ok], Q=Q_ok, p=p_ok).mean().item()
            print(f"Objective mean (converged): {obj_ok_mean:.6f}")

        eP_best, eD_best = problem.primal_dual_infeasibility(
            best_state["x"], best_state["s"],
            best_state["y"], best_state["v"], best_state["z1"], best_state["z2"], best_state["w1"], best_state["w2"],
            best_state["x1E"], best_state["x2E"], best_state["s1E"], best_state["s2E"], best_state["yE"], best_state["vE"], best_state["z1E"], best_state["z2E"], best_state["w1E"], best_state["w2E"],
            best_state["muP"], best_state["muB"], best_state["muA"],
        )
        for name, vals in (("eP", eP_best), ("eD", eD_best)):
            _print_infeas_stats(name, vals)
        if ok.any():
            for name, vals in (("eP", eP_best[ok]), ("eD", eD_best[ok])):
                _print_infeas_stats(f"{name} (converged)", vals)
        else:
            print("No converged problems; skipping converged eP/eD stats.")

    if train_test != "train":
        initial_state = _snapshot_state()
        best_chi = chi.clone()
        best_state = _snapshot_state()
    
    #chi_max = torch.minimum(chi, torch.tensor(1000.0, device=chi.device, dtype=chi.dtype))

    
    
    if (device == 'cpu') and (print_level == 3):
        # display merit/residual
        M_k = problem.merit_M(x, s,      
                y, v, z1, z2, w1, w2,    
                x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
                muP, muB, muA, bad_x1, bad_x2).mean()
        chi = problem.chi(x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA)
        if has_cons:
            print(f"[IPM‐iter {0}] μP={muP.mean().item():.1e}, "
                    f"μB={muB.mean().item():.1e}, μA={muA.mean().item():.1e}, M={M_k:.4e}, ∥r∥={chi_max.mean().item():.4e}")
        else:
            print(f"[IPM‐iter {0}] μB={muB.mean().item():.1e}, μA={muA.mean().item():.1e}, M={M_k:.4e}, ∥r∥={chi_max.mean().item():.4e}")
        
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
 
    if (device == 'cpu') and (print_level == 3):
        dbg_cache = {}
        P = problem.primal_feasibility(x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA, grad_cache=dbg_cache)      
        # dual feasibility
        D = problem.dual_feasibility(x, s,      
                y, v, z1, z2, w1, w2,    
                x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
                muP, muB, muA, grad_cache=dbg_cache)        
        # complementarity
        C = problem.complementarity(x, s,      
                y, v, z1, z2, w1, w2,    
                x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
                muP, muB, muA)   
        print("Primal Feasability: ", P.mean())
        print("Dual Feasability: ", D.mean()) # stationarity is giving me issues
        print("Complimentarity: ", C.mean())

    if (device == 'cpu') and (print_level == 3):
        print("Primal Feasability: ", P.mean())
        for i in range(min(10, lb.shape[0])):
            print(f"Primal Feasability[{i}] = {P.view(B)[i]:.6f}")
        print("Dual Feasability: ", D.mean()) # stationarity is giving me issues
        for i in range(min(10, lb.shape[0])):
            print(f"Dual Feasability[{i}] = {D.view(B)[i]:.6f}")
        for i in range(min(10, lb.shape[0])):
            print(f"Complimentarity[{i}] = {C.view(B)[i]:.6f}")
        # for i in range(min(5, lb.shape[0])):
        #     print(f"Problem {i}:")
        #     for j in range(lb.shape[1]):
        #         print(f"  Variable {j}: z1 = {z1[i, j].item()}, z2 = {z2[i, j].item()}")
        #         print(f"obj_grad[{i}]", problem.obj_grad(x)[i])
        #     print()
        print("Complementarity: ", C.mean())
        
    
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

    

    # outer IPM iterations
    done = None
    if train_test != "train":
        done = torch.zeros(B, dtype=torch.bool, device=device)

    for outer in range(1, max_outer+1):

        
        # flatten for LSTM call
        x = x.squeeze(-1)
        x1 = x1.squeeze(-1)
        x2 = x2.squeeze(-1)
        z1 = z1.squeeze(-1)
        z2 = z2.squeeze(-1)
        if problem.num_ineq > 0:
            s = s.squeeze(-1)
            y = y.squeeze(-1)
            w1 = w1.squeeze(-1)
        if problem.num_eq > 0:
            v = v.squeeze(-1)

        N = 7 * n

        # keep previous state to optionally freeze converged samples
        x_prev = x.clone()
        x1_prev, x2_prev = x1.clone(), x2.clone()
        s_prev  = s.clone()  if problem.num_ineq > 0 else None
        y_prev  = y.clone()  if problem.num_ineq > 0 else None
        v_prev  = v.clone()  if problem.num_eq  > 0 else None
        z1_prev, z2_prev = z1.clone(), z2.clone()
        w1_prev = w1.clone() if problem.num_ineq > 0 else None
        w2_prev = w2.clone() if w2 is not None else None

        # currently does 1 micro-iteration of the LSTM, can change that 
        (x, s, y, v, z1, z2, w1, w2), step_loss_vector, gradient_loss_vector, (lstm_states_x, lstm_states_s, r_k) = net(
            problem,
            x, s, y, v, z1, z2, w1, w2, 
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA, r_k,
            lstm_states_x, lstm_states_s, bad_x1, bad_x2, M_max, n_vec = n_vec, 
            project_step=project_tube_all, outer = outer, print_level = print_level, rollout=rollout
        )

        # freeze already-converged samples (test/val only)
        if train_test == "No" and done is not None and done.any():
            def _pad_trim(t, target_cols):
                if t is None or target_cols is None:
                    return t
                needs_unsq = (t.dim() == 3 and t.shape[-1] == 1)
                if needs_unsq:
                    t = t.squeeze(-1)
                if t.dim() == 1:
                    t = t.unsqueeze(-1)
                if t.shape[1] > target_cols:
                    t = t[:, :target_cols]
                elif t.shape[1] < target_cols:
                    pad = t.new_zeros(t.shape[0], target_cols - t.shape[1])
                    t = torch.cat([t, pad], dim=1)
                if needs_unsq:
                    t = t.unsqueeze(-1)
                return t

            def _ensure_col3(t):
                if t is None:
                    return None
                return t if t.dim() == 3 else t.unsqueeze(-1)

            n_cols = x.shape[1] if x.dim() == 2 else problem.num_var
            m_cols = s.shape[1] if (problem.num_ineq > 0 and s is not None and s.dim() > 1) else getattr(problem, 'num_ineq', None)
            meq_cols = v.shape[1] if (problem.num_eq > 0 and v is not None and v.dim() > 1) else getattr(problem, 'num_eq', None)

            x_prev = _pad_trim(x_prev, n_cols)
            x      = _pad_trim(x, n_cols)
            x1_prev = _pad_trim(x1_prev, n_cols)
            x1      = _pad_trim(x1, n_cols)
            x2_prev = _pad_trim(x2_prev, n_cols)
            x2      = _pad_trim(x2, n_cols)
            z1_prev = _pad_trim(z1_prev, n_cols)
            z1      = _pad_trim(z1, n_cols)
            z2_prev = _pad_trim(z2_prev, n_cols)
            z2      = _pad_trim(z2, n_cols)
            if problem.num_ineq > 0:
                s_prev  = _pad_trim(s_prev, m_cols)
                s       = _pad_trim(s, m_cols)
                y_prev  = _pad_trim(y_prev, m_cols)
                y       = _pad_trim(y, m_cols)
                w1_prev = _pad_trim(w1_prev, m_cols)
                w1      = _pad_trim(w1, m_cols)
            if problem.num_eq > 0:
                v_prev  = _pad_trim(v_prev, meq_cols)
                v       = _pad_trim(v, meq_cols)
            if w2_prev is not None:
                w2_prev = _pad_trim(w2_prev, m_cols)
                w2      = _pad_trim(w2, m_cols)

            # ensure 3D for broadcasting with keep
            x = _ensure_col3(x); x_prev = _ensure_col3(x_prev)
            x1 = _ensure_col3(x1); x1_prev = _ensure_col3(x1_prev)
            x2 = _ensure_col3(x2); x2_prev = _ensure_col3(x2_prev)
            z1 = _ensure_col3(z1); z1_prev = _ensure_col3(z1_prev)
            z2 = _ensure_col3(z2); z2_prev = _ensure_col3(z2_prev)
            if problem.num_ineq > 0:
                s = _ensure_col3(s); s_prev = _ensure_col3(s_prev)
                y = _ensure_col3(y); y_prev = _ensure_col3(y_prev)
                w1 = _ensure_col3(w1); w1_prev = _ensure_col3(w1_prev)
            if problem.num_eq > 0:
                v = _ensure_col3(v); v_prev = _ensure_col3(v_prev)
            if w2_prev is not None:
                w2 = _ensure_col3(w2); w2_prev = _ensure_col3(w2_prev)

            keep = (~done).view(-1, 1, 1)  # [B,1,1]
            x = torch.where(keep, x, x_prev)
            x1 = torch.where(keep, x1, x1_prev) if x1_prev is not None else x1
            x2 = torch.where(keep, x2, x2_prev) if x2_prev is not None else x2
            z1 = torch.where(keep, z1, z1_prev)
            z2 = torch.where(keep, z2, z2_prev)
            if problem.num_ineq > 0:
                s = torch.where(keep, s, s_prev)
                y = torch.where(keep, y, y_prev)
                w1 = torch.where(keep, w1, w1_prev)
            if problem.num_eq > 0:
                v = torch.where(keep, v, v_prev)
            if w2_prev is not None:
                w2 = torch.where(keep, w2, w2_prev)

        #print(step_loss_vector.shape)
        step_loss = step_loss_vector.mean()
        step_loss = step_loss/K_INNER

        gradient_loss = gradient_loss_vector.mean()
        gradient_loss = gradient_loss/K_INNER


        if (device == 'cpu') and (print_level == 3):
            print("Loss for this step: ", step_loss)
            print("Grad loss for this step: ", gradient_loss)
            try:
                obj_now = problem.obj_fn(x).mean().item()
                print("Objective mean for this step: ", obj_now)
                if best_state is not None:
                    x_best = best_state["x"]
                    x_best = x_best.unsqueeze(-1) if x_best.dim() == 2 else x_best
                    best_obj_now = problem.obj_fn(x_best).mean().item()
                    print("Best-state objective mean so far: ", best_obj_now)
            except Exception:
                pass
        if return_info:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start_time_total
            obj_now_vec = problem.obj_fn(x).view(-1).detach().cpu()
            obj_traj_t.append(elapsed)
            obj_traj_v.append(obj_now_vec)

        combined_loss = step_loss + gradient_loss
        if train_test == 'train':
            unroll_losses.append(combined_loss)
        else:
            total_inner_loss = total_inner_loss + combined_loss

        # share expensive gradient evaluations across residual/penalty calculations
        grad_cache = {}

        P = problem.primal_feasibility(x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA, grad_cache=grad_cache) 

        D = problem.dual_feasibility(x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA, grad_cache=grad_cache) 
        C = problem.complementarity(x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA) 
        # use raw (unnormalized) KKT gradients as a stability penalty
        raw_grads = problem.raw_kkt_gradients(
            x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA, bad_x1, bad_x2, grad_cache=grad_cache
        )
        if (device == 'cpu') and (print_level == 3):
            grad_x, grad_s, grad_y, grad_vv, grad_z1, grad_z2, grad_w1, grad_w2 = raw_grads
            print("grad_x: ", grad_x.shape)
            for i in range(min(5, B)):
                print(f"raw grad_x[{i}]: {grad_x[i].mean()}")
                if grad_z1 is not None:
                    print(f"raw grad_z1[{i}]: {grad_z1[i].mean()}")
                if grad_z2 is not None:
                    print(f"raw grad_z2[{i}]: {grad_z2[i].mean()}")
                if grad_vv is not None:
                    print(f"raw grad_vv[{i}]: {grad_vv[i].mean()}")
        # flat_grads = [g.reshape(g.shape[0], -1) for g in raw_grads if g is not None]
        # raw_grad_scale = torch.abs(torch.cat(flat_grads, dim=1)).mean(dim=1)  # [B]
        # total_inner_loss = total_inner_loss + torch.log(raw_grad_scale + 1e-8).mean()

        # Only penalize the x-gradient magnitude for now
        # grad_x_only = raw_grads[0]
        # grad_x_scale = torch.abs(grad_x_only.reshape(grad_x_only.shape[0], -1)).mean(dim=1)
        # total_inner_loss = total_inner_loss + torch.log(grad_x_scale + 1e-8).mean()

        # Penalize each available gradient block separately, then sum their log scales
        # grad_scales = []
        # for g in raw_grads:
        #     if g is None:
        #         continue
        #     # flatten over all dims except batch before measuring scale
        #     g_flat = g.reshape(g.shape[0], -1)
        #     # switch to mean absolute gradient magnitude across all parameters
        #     #grad_scales.append(g_flat.abs().mean(dim=1))  # [B] mean |g|
        #     grad_scales.append(torch.norm(g_flat, dim=1).pow(2))   # [B]  (L2)
        #     #grad_scales.append(g_flat.abs().amax(dim=1))     # [B]  (L∞)
        # if grad_scales:
        #     # sum all grad magnitudes per sample first, then take one log
        #     sum_per_sample = torch.stack(grad_scales, dim=0).sum(dim=0)  # [B]
        #     total_inner_loss = total_inner_loss + torch.log(sum_per_sample + 1e-8).mean()
        
        #total_inner_loss = total_inner_loss/n

        # M_parts = problem.Mtest(x, s,      
        #     y, v, z1, z2, w1, w2,    
        #     x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
        #     muP, muB, muA, M_max, bad_x1, bad_x2)   
        
        # # Mtest returns tuple: (M_x, M_s, M_z1, M_z2, M_w1, M_y, M_v, aggregate)
        # if isinstance(M_parts, (tuple, list)) and len(M_parts) >= 7:
        #     M_x, M_s, M_z1, M_z2, M_w1, M_y, M_v, *_ = M_parts
        #     if (device == 'cpu') and (print_level == 3):
        #         for i in range(min(5, B)):
        #             print(f"M_x[{i}]: {M_x[i].item() if M_x.dim()==1 else M_x[i].tolist()}")
        #             print(f"M_z1[{i}]: {M_z1[i].item() if M_z1.dim()==1 else M_z1[i].tolist()}")
        #             print(f"M_z2[{i}]: {M_z2[i].item() if M_z2.dim()==1 else M_z2[i].tolist()}")
        #             if M_v is not None:
        #                 print(f"M_v[{i}]: {M_v[i].item() if M_v.dim()==1 else M_v[i].tolist()}")
        #     logs = []
        #     for term in (M_x, M_s, M_z1, M_z2, M_w1, M_y, M_v):
        #         if term is None:
        #             continue
        #         if torch.is_tensor(term) and torch.all(term == 0):
        #             continue
        #         logs.append(torch.log(term + 1e-8))
        #     terms = torch.stack(logs, dim=0)   # [num_terms, B, …]
        #     # print(terms.shape)
        #     # print(terms.sum(dim=0).shape)
        #     total_inner_loss = terms.sum(dim=0).mean()  # sum per sample, then average over batch

        # else:
        #     total_inner_loss = total_inner_loss + (torch.log(M_parts + 1e-8)).mean()

        #total_inner_loss = total_inner_loss + (torch.log(P + 1e-8) + torch.log(D+ 1e-8) + torch.log(C+ 1e-8)).mean()

        #total_inner_loss = total_inner_loss + (P + D + C).mean()

        #total_inner_loss = total_inner_loss/n

        # MU_TARGET = 1e-4      # e.g. 1e-4 or 1e-5

        # import math
        # import torch.nn.functional as F

        # # shapes: muB, muP are [B] or [B,1]
        # muB_flat = muB.view(-1)
        # muP_flat = muP.view(-1)

        # log_muB = torch.log10(muB_flat.clamp_min(1e-12))
        # log_muP = torch.log10(muP_flat.clamp_min(1e-12))
        # log_target = math.log10(MU_TARGET)

        # # positive part of (log μ - log μ_target)
        # excess_B = F.relu(log_muB - log_target)
        # excess_P = F.relu(log_muP - log_target)

        # # e.g. squared penalty on excess decades
        # L_mu = (excess_B**2 + excess_P**2).mean()

        # res = P + D + C      # [B]

        # RES_GATE = 1e-2    # when res < 1e-2, we start caring about μ more
        # KAPPA    = 5e-3    # softness of the transition

        # # detach so the gating doesn't backprop through residuals
        # w_mu = torch.sigmoid((RES_GATE - res.detach()) / KAPPA)   # [B] in (0,1)

        # # apply weights to the μ-penalty per sample
        # L_mu_weighted = (w_mu * (excess_B**2 + excess_P**2)).sum() / (w_mu.sum() + 1e-8)

        # lambda_mu = 0.1   # tune this

        # L_res = res.mean()               
        # #loss  = L_res + lambda_mu * L_mu_weighted



        # total_inner_loss = total_inner_loss + (lambda_mu * L_mu_weighted)
        
        #total_inner_loss = P.sum() + D.sum() + C.sum()
             
        
        # if train_test == 'train':
        #     meta_opt.zero_grad(set_to_none=True)      # clear old gradients
        #     step_loss.backward()                      # back-prop through the micro step
        #     torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        #     meta_opt.step()                           # update the LSTM weights
        # else:
        #     val_flag = False

        x1E = x1E.detach(); x2E = x2E.detach();z1E = z1E.detach();z2E = z2E.detach()
        if problem.num_ineq > 0:
            s1E = s1E.detach(); yE = yE.detach(); w1E = w1E.detach()

        if problem.num_eq > 0:
            vE = vE.detach()
        muB = muB.detach(); muP = muP.detach(); muA = muA.detach()

        bad_x1 = bad_x1.detach()
        bad_x2 = bad_x2.detach()
        M_max = M_max.detach()

        detach_prob = getattr(net, "detach_prob", 0.0)
        step_in_window = ((outer - 1) % BACKPROP_EVERY) + 1

        def _detach_state():
            nonlocal x, s, y, v, z1, z2, w1, w2, x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, muB, muP, muA, r_k, lstm_states_x, lstm_states_s, bad_x1, bad_x2, M_max, total_inner_loss
            x = x.detach()
            z1 = z1.detach()
            z2 = z2.detach()
            if problem.num_eq > 0 and v is not None:
                v = v.detach()
            if problem.num_ineq > 0:
                if y is not None:
                    y = y.detach()
                if w1 is not None:
                    w1 = w1.detach()
                if s is not None:
                    s = s.detach()
            if w2 is not None:
                w2 = w2.detach()
            x1E = x1E.detach()
            x2E = x2E.detach()
            z1E = z1E.detach()
            z2E = z2E.detach()
            if s1E is not None:
                s1E = s1E.detach()
            if s2E is not None:
                s2E = s2E.detach()
            if yE is not None:
                yE = yE.detach()
            if vE is not None:
                vE = vE.detach()
            if w1E is not None:
                w1E = w1E.detach()
            if w2E is not None:
                w2E = w2E.detach()
            muB = muB.detach()
            muP = muP.detach()
            muA = muA.detach()
            bad_x1 = bad_x1.detach()
            bad_x2 = bad_x2.detach()
            if M_max is not None:
                M_max = M_max.detach()
            if lstm_states_x is not None:
                h_x, c_x = lstm_states_x
                h_x = h_x.detach() if h_x is not None else None
                c_x = c_x.detach() if c_x is not None else None
                lstm_states_x = (h_x, c_x)
            if lstm_states_s is not None:
                h_s, c_s = lstm_states_s
                h_s = h_s.detach() if h_s is not None else None
                c_s = c_s.detach() if c_s is not None else None
                lstm_states_s = (h_s, c_s)
            if r_k is not None:
                r_k = r_k.detach()

        if train_test == 'train':
            if detach_prob > 0 and step_in_window == detach_next:
                _detach_state()

            if outer % BACKPROP_EVERY == 0:
                if unroll_losses:
                    losses_tensor = torch.stack(unroll_losses)
                    full_loss = losses_tensor.sum()
                    steps = losses_tensor.shape[0]
                    n_prog = random.randint(0, steps - 1)
                    k_prog = random.randint(1, steps - n_prog)
                    prog_steps = max(1, min(steps, n_prog + k_prog))
                    prog_loss = losses_tensor[:prog_steps].sum()
                    alpha_blend = getattr(net, "alpha", 0.8)
                    blended_loss = (1 - alpha_blend) * full_loss + alpha_blend * prog_loss
                else:
                    blended_loss = total_inner_loss

                meta_opt.zero_grad(set_to_none=True)
                blended_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                meta_opt.step()

                total_inner_loss = 0
                unroll_losses = []

                detach_next = random.randint(1, BACKPROP_EVERY)
                _detach_state()

        
        chi = (P + D + C).view(B)
        if train_test != "train":
            improved = chi < best_chi
            if improved.any():
                best_chi = torch.where(improved, chi, best_chi)
                best_state["x"] = _update_best_tensor(best_state["x"], x, improved)
                best_state["s"] = _update_best_tensor(best_state["s"], s, improved)
                best_state["y"] = _update_best_tensor(best_state["y"], y, improved)
                best_state["v"] = _update_best_tensor(best_state["v"], v, improved)
                best_state["z1"] = _update_best_tensor(best_state["z1"], z1, improved)
                best_state["z2"] = _update_best_tensor(best_state["z2"], z2, improved)
                best_state["w1"] = _update_best_tensor(best_state["w1"], w1, improved)
                best_state["w2"] = _update_best_tensor(best_state["w2"], w2, improved)
                best_state["x1E"] = _update_best_tensor(best_state["x1E"], x1E, improved)
                best_state["x2E"] = _update_best_tensor(best_state["x2E"], x2E, improved)
                best_state["s1E"] = _update_best_tensor(best_state["s1E"], s1E, improved)
                best_state["s2E"] = _update_best_tensor(best_state["s2E"], s2E, improved)
                best_state["yE"] = _update_best_tensor(best_state["yE"], yE, improved)
                best_state["vE"] = _update_best_tensor(best_state["vE"], vE, improved)
                best_state["z1E"] = _update_best_tensor(best_state["z1E"], z1E, improved)
                best_state["z2E"] = _update_best_tensor(best_state["z2E"], z2E, improved)
                best_state["w1E"] = _update_best_tensor(best_state["w1E"], w1E, improved)
                best_state["w2E"] = _update_best_tensor(best_state["w2E"], w2E, improved)
                best_state["muP"] = _update_best_tensor(best_state["muP"], muP, improved)
                best_state["muB"] = _update_best_tensor(best_state["muB"], muB, improved)
                best_state["muA"] = _update_best_tensor(best_state["muA"], muA, improved)
            # revert regressions relative to initialization
            # regressed = chi > 3*initial_chi
            # if regressed.any():
            #     (x, x1, x2, z1, z2,
            #      s, s1, w1, y, v,
            #      muB, muP,
            #      x1E, x2E, s1E, yE, vE, z1E, z2E, w1E) = _restart_from_init(
            #         regressed, initial_state, noise_scale=0.05, lb=lb_vals, ub=ub_vals
            #     )
            #     # recalc metrics after revert
            #     P = problem.primal_feasibility(x, s,
            #         y, v, z1, z2, w1, w2,
            #         x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E,
            #         muP, muB, muA)
            #     D = problem.dual_feasibility(x, s,
            #         y, v, z1, z2, w1, w2,
            #         x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E,
            #         muP, muB, muA)
            #     C = problem.complementarity(x, s,
            #         y, v, z1, z2, w1, w2,
            #         x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E,
            #         muP, muB, muA)
            #     chi = (P + D + C).view(B)

        if device == 'cpu':
            # display merit/residual
            M_k_new = problem.merit_M(x, s,      
                y, v, z1, z2, w1, w2,    
                x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
                muP, muB, muA, bad_x1, bad_x2).mean().item()

            # f_val, lb_1, lb_2, lb_3, lb_4, AL_1, ub_1, ub_2, ub_3, ub_4, AL_2 = problem.merit_M_indi(x, s,      
            #     y, v, z1, z2, w1, w2,    
            #     x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            #     muP, muB, muA, bad_x1, bad_x2)
            
            # if device == 'cpu':
            #     print("f_val: ", f_val.mean().item())
            #     print("lb_1: ", lb_1.mean().item())
            #     print("lb_2: ", lb_2.mean().item())
            #     print("lb_3: ", lb_3.mean().item())
            #     print("lb_4: ", lb_4.mean().item())
            #     print("AL_1: ", AL_1.mean().item())
            #     print("ub_1: ", ub_1.mean().item())
            #     print("ub_2: ", ub_2.mean().item())
            #     print("ub_3: ", ub_3.mean().item())
            #     print("ub_4: ", ub_4.mean().item())
            #     print("AL_2: ", AL_2.mean().item())

            
            if print_level == 3:
                obj_fcn = problem.obj_fn(x).mean().item()

                chi = problem.chi(x, s,      
                    y, v, z1, z2, w1, w2,    
                    x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
                    muP, muB, muA)
                if has_cons:
                    print(f"[IPM‐iter {outer}] μP={muP.mean().item():.1e}, "
                        f"μB={muB.mean().item():.1e}, μA={muA.mean().item():.1e}, M={M_k_new:.4e}, ∥r∥={chi.mean().item():.4e}")
                else:
                    print(f"[IPM‐iter {outer}] μB={muB.mean().item():.1e}, μA={muA.mean().item():.1e}, M={M_k_new:.4e}, ∥r∥={chi.mean().item():.4e}")
                
                print("x: ", x.mean().item())
                print("z1: ", z1.mean().item())
                print("z2: ", z2.mean().item())

                for i in range(min(5, lb.shape[0])):
                    print(f"obj[{i}] = {problem.obj_fn(x).view(B)[i]:.6f}")
       
  
        if (device == 'cpu') and (print_level == 3):
            print("Primal Feasability: ", P.mean())
            for i in range(min(10, lb.shape[0])):
                print(f"Primal Feasability[{i}] = {P.view(B)[i]:.6f}")
            print("Dual Feasability: ", D.mean()) # stationarity is giving me issues
            for i in range(min(10, lb.shape[0])):
                print(f"Dual Feasability[{i}] = {D.view(B)[i]:.6f}")
            for i in range(min(10, lb.shape[0])):
                print(f"Complimentarity[{i}] = {C.view(B)[i]:.6f}")
            # for i in range(min(5, lb.shape[0])):
            #     print(f"Problem {i}:")
            #     for j in range(lb.shape[1]):
            #         print(f"  Variable {j}: z1 = {z1[i, j].item()}, z2 = {z2[i, j].item()}")
            #         print(f"obj_grad[{i}]", problem.obj_grad(x)[i])
            #     print()
            print("Complementarity: ", C.mean())
        if (device == 'cpu') and (print_level == 3):
            if ((D.mean() < 0.1) & (P.mean() < 0.1) & (C.mean() < 0.1)):
                print("CONVERGED ")
            thresh = 1e-2
            eP, eD = problem.primal_dual_infeasibility(
                x, s,
                y, v, z1, z2, w1, w2,
                x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E,
                muP, muB, muA
            )
            print("eP: ", eP.mean())
            print("eD: ", eD.mean())
            #ok = (P < thresh) & (D < thresh) & (C < thresh)   # elementwise
            ok = (eP < thresh) & (eD < thresh)
            print("Percentage converged: ", sum(ok)/B)
            if return_info:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = time.perf_counter() - start_time_total
                newly = (conv_time == float("inf")) & ok.to(conv_time.device)
                if newly.any():
                    conv_time = torch.where(newly, torch.full_like(conv_time, elapsed), conv_time)
                last_ok = ok
            # do not early-return; continue full unroll


        # early stopping mechanism
        if train_test != "train":
            thresh = 1e-2
            # eP, eD = problem.primal_dual_infeasibility(
            #     x, s,
            #     y, v, z1, z2, w1, w2,
            #     x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E,
            #     muP, muB, muA
            # )
            # print("eP: ", eP.mean())
            # print("eD: ", eD.mean())
            ok = (P < thresh) & (D < thresh) & (C < thresh)  # elementwise
            if return_info:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = time.perf_counter() - start_time_total
                newly = (conv_time == float("inf")) & ok.to(conv_time.device)
                if newly.any():
                    conv_time = torch.where(newly, torch.full_like(conv_time, elapsed), conv_time)
                last_ok = ok
            done = done | ok
            if ok.all().item():   # True iff ALL problems pass
                if (device == 'cpu'):
                    print("Objective value: ", problem.obj_fn(x).mean())
                    print("Primal Feasability: ", P.mean())
                    print("Dual Feasability: ", D.mean())
                    print("Complementarity: ", C.mean())
                _print_best_state_metrics()
                if return_info:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    elapsed = time.perf_counter() - start_time_total
                    newly = (conv_time == float("inf")) & ok.to(conv_time.device)
                    if newly.any():
                        conv_time = torch.where(newly, torch.full_like(conv_time, elapsed), conv_time)
                    last_ok = ok
                # do not early-return; continue full unroll

        mask_lb_3d = mask_lb.unsqueeze(-1) if mask_lb.dim() == 2 else mask_lb  # [B,n,1]
        mask_ub_3d = mask_ub.unsqueeze(-1) if mask_ub.dim() == 2 else mask_ub

        raw_x1 = x - lb_vals.unsqueeze(-1)        # [B,n,1]
        x1     = torch.where(mask_lb_3d, raw_x1, torch.zeros_like(raw_x1))

        raw_x2 = ub_vals.unsqueeze(-1) - x        # [B,n,1]
        x2     = torch.where(mask_ub_3d, raw_x2, torch.zeros_like(raw_x2))
        
        if problem.num_ineq > 0:
            s1 = s.clone()

        _, _, _, _, _, _, _, Mtest = problem.Mtest(x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA, M_max, bad_x1, bad_x2, raw_grads=raw_grads, grad_cache=grad_cache)   

        mask_O = chi < chi_max               # [B]  O-iterate candidates
        
        mask_M = (~mask_O) & (Mtest < tolM)  # M if not O

        if (device == 'cpu') & (print_level == 3):
            print("Mtest: ", Mtest[0])
            print("TolM: ",tolM[0])

        # something wrong with Mtest condition

        mask_F = ~(mask_O | mask_M)          # the rest

        mask_lb = torch.isfinite(lb_vals) if lb_vals is not None else None
        mask_ub = torch.isfinite(ub_vals)   if ub_vals is not None else None

        mask_lb = mask_lb.unsqueeze(-1)
        mask_ub = mask_ub.unsqueeze(-1)


        x1E_new, x2E_new = x1E.clone(), x2E.clone()     # [B,n,1]
        z1E_new, z2E_new = z1E.clone(), z2E.clone()     # [B,n,1]
        if problem.num_ineq > 0:
            s1E_new = s1E.clone(); w1E_new = w1E.clone()
            yE_new = y.clone()#; vE_new = vE.clone()
        if problem.num_eq > 0:
            vE_new = vE.clone()
        chi_max_new      = chi_max.clone()              # [B]
        tolM_new         = tolM.clone()                 # [B]
        muB_new          = muB.squeeze(-1).clone()                  # [B]
        muP_new          = muP.squeeze(-1).clone()  
        muA_new          = muA.squeeze(-1).clone()  

        muB = muB.squeeze(-1)
        muP = muP.squeeze(-1)
        muA = muA.squeeze(-1)

        #print("Mtest: ", sum(mask_M))

        if mask_O.any():
            #print("O-iterate")
            # 1) reduce chi_max by half

            chi_max_new[mask_O] /= 2
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
            if problem.num_ineq > 0:
                s1E_new[mask_O] = s1[mask_O]
                w1E_new[mask_O] = w1[mask_O]
                yE_new[mask_O] = y[mask_O]
            if problem.num_eq > 0:
                vE_new[mask_O] = v[mask_O]

        if mask_M.any():
            #print("Mtest: ", sum(mask_M))
            #print("M-iterate")
            #1) clamp (inequality part) of slacks & multipliers
            x1E_new[mask_M] = x1[mask_M].clamp(0.0, x1_max_e)
            x2E_new[mask_M] = x2[mask_M].clamp(0.0, x2_max_e)
            z1E_new[mask_M] = z1[mask_M].clamp(0.0, z1_max_e)
            z2E_new[mask_M] = z2[mask_M].clamp(0.0, z2_max_e)
            if problem.num_ineq > 0:
                s1E_new[mask_M] = s1E[mask_M].clamp(0.0, s1_max_e)
                w1E_new[mask_M] = w1E[mask_M].clamp(0.0, w1_max_e)
                yE_new[mask_M] = yE[mask_M].clamp(0.0, y_max_e)
            if problem.num_eq > 0:
                vE_new[mask_M] = vE[mask_M].clamp(0.0, v_max_e)

            # # make tolM broadcastable to [B_M, 1, 1]
            tolM_M = tolM_new[mask_M].view(-1, 1, 1)          # [B_M,1,1]

            tolM_new[mask_M] = tolM[mask_M] * 0.5

            #muB_new[(mask_M) & (muB_new > 1e-10)] = muB_new[(mask_M) & (muB_new > 1e-10)]/2
            #muB_new[~mask_M] = muB[~mask_M]
            #if train_test != "train":
            if outer > 0:

                # 1) complementarity check (C already [B])
                bad_comp = C[mask_M] > tolM_M.squeeze(-1).squeeze(-1)           # [B_M]

                # 2) slack & multiplier sign checks
                bad_x1 = (x1[mask_M] < -tolM_M).any(dim=2).any(dim=1)   # [B_M]
                bad_x2 = (x2[mask_M] < -tolM_M).any(dim=2).any(dim=1)
                bad_z1 = (z1[mask_M] < -tolM_M).any(dim=2).any(dim=1)
                bad_z2 = (z2[mask_M] < -tolM_M).any(dim=2).any(dim=1)

                if problem.num_ineq > 0:
                    bad_s1 = (s1[mask_M] < -tolM_M).any(dim=2).any(dim=1)
                    bad_w1 = (w1[mask_M] < -tolM_M).any(dim=2).any(dim=1)

                # aggregate “bad” flag  (logical OR)
                if problem.num_ineq > 0:
                    shrink_muB = bad_comp | bad_x1 | bad_x2 | bad_z1 | bad_z2 | bad_s1 | bad_w1
                else:
                    shrink_muB = bad_comp | bad_x1 | bad_x2 | bad_z1 | bad_z2   # [B_M]

                # keep if NOT bad
                keep_muB  = ~shrink_muB                                     # [B_M]

                # apply update:  μB ← μB / μBfac   where shrink flag is True
                muB_new[mask_M] = torch.where(keep_muB,
                                            muB[mask_M],
                                            muB[mask_M]/2)
                
                # # 1) Primal check
                # bad_primal = P[mask_M] > tolM_M.squeeze(-1).squeeze(-1)           # [B_M]

                # muP_new[mask_M] = torch.where(bad_primal,
                #                             muP[mask_M]/2,
                #                             muP[mask_M])

                if (problem.num_ineq > 0) or (problem.num_eq > 0):
                    bad_primal = P[mask_M] > tolM_M.squeeze(-1).squeeze(-1)           # [B_M]
                    muP_new[mask_M] = torch.where(bad_primal,
                                                muP[mask_M]/2,
                                                muP[mask_M])
                else:
                    muP_new[mask_M] = muP[mask_M]
            


            

        PERIOD = 300
        OFF_B, OFF_P = 0, 150

        if outer < 0:
            if outer % PERIOD == OFF_B:  
                muB_new *= 0.5
            if outer % PERIOD == OFF_P:   
                muP_new *= 0.5
            # if outer % PERIOD == OFF_A:   
            #     muA_new *= 0.5


        tolM  = tolM.clone()        # ← breaks the broadcast view
        chi_max = chi_max.clone()
        muB  = muB.clone()
        muP  = muP.clone()
        muA  = muA.clone()

        # # freeze converged samples: keep old solver state
        # if train_test != "train" and done is not None and done.any():
        #     keep_mask = (~done)
        #     keep_mask_col = keep_mask.view(-1, 1, 1)
        #     chi_max_new = torch.where(keep_mask, chi_max_new, chi_max)
        #     tolM_new    = torch.where(keep_mask, tolM_new, tolM)
        #     muB_new     = torch.where(keep_mask, muB_new, muB)
        #     muP_new     = torch.where(keep_mask, muP_new, muP)
        #     muA_new     = torch.where(keep_mask, muA_new, muA)

        #     x1E_new = torch.where(keep_mask_col, x1E_new, x1E)
        #     x2E_new = torch.where(keep_mask_col, x2E_new, x2E)
        #     z1E_new = torch.where(keep_mask_col, z1E_new, z1E)
        #     z2E_new = torch.where(keep_mask_col, z2E_new, z2E)
        #     if problem.num_ineq > 0:
        #         s1E_new = torch.where(keep_mask_col, s1E_new, s1E)
        #         w1E_new = torch.where(keep_mask_col, w1E_new, w1E)
        #         yE_new  = torch.where(keep_mask_col, yE_new, yE)
        #     if problem.num_eq > 0:
        #         vE_new  = torch.where(keep_mask_col, vE_new, vE)

        # Commit updated values back to the solver state
        chi_max.copy_(chi_max_new)
        tolM.copy_(tolM_new)
        muB.copy_(muB_new)
        muP.copy_(muP_new)
        muA.copy_(muA_new)
        muB = muB.unsqueeze(-1)
        muP = muP.unsqueeze(-1)
        muA = muA.unsqueeze(-1)

        x1E.copy_(x1E_new)
        x2E.copy_(x2E_new)
        z1E.copy_(z1E_new)
        z2E.copy_(z2E_new)
        if problem.num_ineq > 0:
            s1E.copy_(s1E_new)
            w1E.copy_(w1E_new)
            yE.copy_(yE_new)
        if problem.num_eq > 0:
            vE = vE_new.detach().clone()


        slack1 = x1.squeeze(-1) + muB        # should stay > 0
        slack2 = x2.squeeze(-1) + muB        # should stay > 0

        # 2) build a mask of infeasible entries
        bad_x1 = (slack1 <= 0)   # shape [B,n,1], True wherever x1+μB ≤ 0
        bad_x2 = (slack2 <= 0)   # shape [B,n,1], True wherever x2+μB ≤ 0

        slackz1 = (z1.squeeze(-1) + muB)    # [B,n]
        slackz2 = (z2.squeeze(-1) + muB)    # [B,n]

        # build masks of the exact indices that went non-positive
        bad_z1 = (slackz1 <= 0).unsqueeze(-1)  # [B,n,1]
        bad_z2 = (slackz2 <= 0).unsqueeze(-1)  # [B,n,1]

        # Option A: out-of-place with torch.where
        z1 = torch.where(bad_z1, torch.zeros_like(z1), z1)
        z2 = torch.where(bad_z2, torch.zeros_like(z2), z2)

        if problem.num_ineq > 0:
            slackw1 = (w1.squeeze(-1) + muB)    # [B,n]
            slacks1 = (s1.squeeze(-1) + muB)    # [B,n]

            # build masks of the exact indices that went non-positive
            bad_s1 = (slacks1 <= 0).unsqueeze(-1)  # [B,n,1]
            bad_w1 = (slackw1 <= 0).unsqueeze(-1)  # [B,n,1]

            # Option A: out-of-place with torch.where
            s = torch.where(bad_s1, torch.zeros_like(s), s)
            w1 = torch.where(bad_w1, torch.zeros_like(w1), w1)


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
        if (device == 'cpu'):
            M_k = M_k_new
    _print_best_state_metrics()
    # fix the eP and eD: Done
    # get out the mean, max, (median) of eP, eD for the best states and report it
    # Report the time in comparison to IPOPT
    # clean the code for more efficiency
    # Randomize hidden state for the model to try a different direction?
    # Train on 50 iterations?

    # nonconvex, equality constraints training
    # print("Loss for this step: ", step_loss_vector.mean())
    # print("Objective value: ", problem.obj_fn(x).mean())
    # print("Primal Feasability: ", P.mean())
    # print("Dual Feasability: ", D.mean()) # stationarity is giving me issues
    # print("Complementarity: ", C.mean())
    # thresh = 1e-2

    thresh = 1e-2
    eP, eD = problem.primal_dual_infeasibility(
        x, s,
        y, v, z1, z2, w1, w2,
        x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E,
        muP, muB, muA
    )
    print("eP: ", eP.mean())
    print("eD: ", eD.mean())
    #ok = (P < thresh) & (D < thresh) & (C < thresh)   # elementwise
    ok = (eP < thresh) & (eD < thresh)
    print("Percentage converged: ", sum(ok)/B)
    if return_info:
        x_plain = x.squeeze(-1) if x.dim() == 3 else x
        x1_bv = problem.lower_bound_dist(x_plain)
        x2_bv = problem.upper_bound_dist(x_plain)
        bc_violation = torch.maximum(x1_bv, x2_bv).flatten(1).norm(p=2, dim=1)
        return {
            "iters": outer,
            "obj": problem.obj_fn(x).mean().item(),
            "P_mean": P.mean().item(),
            "D_mean": D.mean().item(),
            "C_mean": C.mean().item(),
            "converged_frac": float((last_ok if last_ok is not None else ok).sum().item()) / float(B),
            "ok_mask": (last_ok if last_ok is not None else ok).detach().cpu(),
            "conv_time": conv_time.detach().cpu() if conv_time is not None else None,
            "obj_per_sample": problem.obj_fn(x).view(-1).detach().cpu(),
            "bc_violation": bc_violation.detach().cpu(),
            "obj_traj_times": obj_traj_t,
            "obj_traj_vals": obj_traj_v,
        }
    return outer
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
    p.add_argument("--mode", choices=["train", "test", "profile", "model_comparison"], default="train")
    p.add_argument("--ckpt", default="l20_lstm_ipm.pt")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--ipmlstm_times", type=str, default=None, help="Path to JSON or newline-separated txt with IPM-LSTM batch times (seconds) for performance profile.")
    p.add_argument("--profile_png", type=str, default="performance_profile.png", help="Where to save performance profile plot.")
    p.add_argument("--profile_csv", type=str, default="performance_profile.csv", help="Where to save per-batch timing table.")
    p.add_argument("--hidden_size", type=int, default=128, help="Hidden size for PS_L20_LSTM.")
    p.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers for PS_L20_LSTM.")
    p.add_argument("--backprop", type=int, default=1, help="Number of outer iterations between backprop steps (BACKPROP_EVERY).")
    p.add_argument("--compare_models", nargs="*", default=None, help="List of additional learned-optimizer ckpt paths for model comparison mode.")
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

    # mat_names = ["QP_convex_10var_0eq_0ineq",
    #              "QP_convex_30var_0eq_0ineq", 
    #              "QP_convex_50var_0eq_0ineq", 
    #              "QP_convex_70var_0eq_0ineq", 
    #              "QP_convex_90var_0eq_0ineq", ]

    mat_names = ["QP_convex_10var_0eq_0ineq"]

    # mat_names = [
    #     "portfolio_QP_box_s50_t5_eq6"
    # ]

    # mat_names = [
    #     "portfolio_QP_box_s10_t2_eq3"
    # ]


    #mat_names = ["QP_convex_100var_50eq_0ineq"]

    #mat_names = ["unconstrained_QP_nonconvex_full_Q_dim_200"]

    #
    # mat_names = [
    #     #"QP_convex_5var_2eq_0ineq",
    #     "QP_convex_10var_5eq_0ineq",
    #     # "QP_convex_20var_10eq_0ineq",
    #     #"QP_convex_30var_15eq_0ineq"
    #     # "QP_convex_40var_20eq_0ineq",
    #     # "QP_convex_50var_25eq_0ineq",
    #     # "QP_convex_60var_30eq_0ineq"
    # ]

    # mat_names = ['QP_convex_12var_3eq_0ineq']


    file_paths = [os.path.join("datasets", "qp", f"{nm}.mat") for nm in mat_names]

    # 2) build pools for each dataset
    pools = []
    for fp in file_paths:
        pools.append({
            "train": QP(prob_type='QP_unconstrained', learning_type='train', file_path=fp, seed=args.seed),
            "val":   QP(prob_type='QP_unconstrained', learning_type='val',   file_path=fp, seed=args.seed),
            "test":  QP(prob_type='QP_unconstrained', learning_type='test',  file_path=fp, seed=args.seed),
        })

    # move all pool tensors to the target device once to avoid per-batch host↔device traffic
    def move_pool_to_device(pool, device):
        N = None
        for name, val in list(pool.__dict__.items()):
            if isinstance(val, torch.Tensor):
                pool.__dict__[name] = val.to(device)
                if N is None and val.dim() >= 1:
                    N = val.size(0)
            elif isinstance(val, np.ndarray):
                tval = torch.from_numpy(val).to(device)
                pool.__dict__[name] = tval
                if N is None and tval.dim() >= 1:
                    N = tval.size(0)
        return N

    def slice_pool(pool, idxs, N):
        mini = copy.copy(pool)
        for name, val in pool.__dict__.items():
            if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.size(0) == N:
                mini.__dict__[name] = val.index_select(0, idxs)
        return mini

    # 3) init net using the first train pool (shapes/masks assumed consistent with padding)
    net = PS_L20_LSTM(
        problem=pools[0]["train"],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        K_inner=K_INNER,
        device=DEVICE
    ).to(DEVICE)

    # training loop
    if args.mode == "train":

        meta_opt = optim.Adam(net.parameters(), lr=LR_META) # adam optimizer
        BATCH = 256
        #print("N: ", N)
        for ep in range(1, EPOCHS+1):
            #print("No chi")
            print("EPOCH: ", ep)
            ds_order = torch.randperm(len(pools)).tolist()

            for di, ds_idx in enumerate(ds_order, 1):
                train_pool = pools[ds_idx]["train"]
                N = move_pool_to_device(train_pool, DEVICE)
                print(f"  DATASET {di}/{len(pools)}: {mat_names[ds_idx]}")
                t0 = time.time()
                N  = train_pool.Q.shape[0]
                perm = torch.arange(N, device=train_pool.Q.device)
                for i in range(0, N, BATCH):
                    batch_idx = i // BATCH + 1
                    num_batches = (N + BATCH - 1) // BATCH  # ceil div without math
                    print(f"BATCH: {batch_idx}/{num_batches}")
                    idxs = perm[i : i + BATCH].to(DEVICE).long()
                    mini = slice_pool(train_pool, idxs, N)

                    _ = solve_one_qcqp(meta_opt, mini, net, backprop_every=args.backprop)
                    # get the objective 
                    # evaluate(val_pool, net, tag="val")
                print(f"Epoch {ep:3d} | {time.time()-t0:.1f}s")
                # if flag:
                #     break
            #solve_one_qcqp(meta_opt, pools[0]['val'], net, train_test = 'val')
        torch.save(net.state_dict(), args.ckpt)
        print(f"✓ weights saved to {args.ckpt}")

    # test evaluation 
    elif args.mode == "test":
        with torch.no_grad():
            import time
            net.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
            print(f"✓ loaded weights from {args.ckpt}")
            net.to(DEVICE).eval()

            import copy
            import numpy as np

            BATCH = 10000  # set your eval batch size
            total_time_all = 0.0  # total time over all datasets

            for ds_idx, pack in enumerate(pools):
                test_pool = pack["val"]   # or pack["test"] if you want to eval on test
                N = move_pool_to_device(test_pool, DEVICE)
                N = test_pool.Q.shape[0]
                print(f"\nTesting: {mat_names[ds_idx]} | N={N} | batch={BATCH}")

                # a dummy optimizer if solve_one_qcqp expects one (won't be stepped in 'test' mode)
                dummy_opt = optim.Adam(net.parameters(), lr=LR_META)

                perm = torch.arange(N, device=test_pool.Q.device)
                num_batches = (N + BATCH - 1) // BATCH

                iterations = []
                ds_start = time.perf_counter()  # start timing this dataset
                ok_mask_full = torch.zeros(N, dtype=torch.bool)

                for i in range(0, N, BATCH):
                    b = i // BATCH + 1
                    print(f"BATCH: {b}/{num_batches}")
                    idxs = perm[i:i+BATCH].to(DEVICE).long()
                    mini = slice_pool(test_pool, idxs, N)

                    # run one batched solve in test mode (no learning updates inside)
                    batch_start = time.perf_counter()
                    info = solve_one_qcqp(dummy_opt, mini, net, train_test='test', return_info=True, backprop_every=args.backprop)
                    batch_end = time.perf_counter()

                    batch_iters = info["iters"] if isinstance(info, dict) and "iters" in info else info
                    iterations.append(batch_iters)
                    if isinstance(info, dict) and "ok_mask" in info and info["ok_mask"] is not None:
                        ok_mask_batch = info["ok_mask"].view(-1).cpu().bool()
                        ok_mask_full[idxs.cpu()] = ok_mask_batch
                    print(f"  batch time: {batch_end - batch_start:.4f} s, iterations: {batch_iters}")

                ds_elapsed = time.perf_counter() - ds_start
                total_time_all += ds_elapsed

                avg_iters = sum(iterations) / len(iterations)
                print(f"Average number of iterations: {avg_iters}")
                print(f"Total time for dataset {mat_names[ds_idx]}: {ds_elapsed:.4f} s")

                # ----------------------------------------------------
                # Ipopt baseline (box + equality + inequality constraints)
                # using your extended `ipopt_box_qp` solver_type
                # ----------------------------------------------------
                # Here we run Ipopt once on the test split of the same dataset.
                sols, _, para_times, ip_iters, bc_violation = pack["val"].opt_solve(
                    solver_type='ipopt_box_qp_extended',
                    tol=1e-2
                )

                # Evaluate objective of the Ipopt solutions
                sols_t = torch.tensor(sols).unsqueeze(-1).float().to(DEVICE)
                best_obj = pack["val"].obj_fn(sols_t).mean().item()
                # same objective but restricted to problems that our learned optimizer converged on
                if ok_mask_full.any():
                    mask_dev = ok_mask_full.to(sols_t.device)
                    sols_masked = sols_t[mask_dev]
                    best_obj_masked = pack["val"].obj_fn(
                        sols_masked,
                        Q=pack["val"].Q[mask_dev] if hasattr(pack["val"], "Q") else None,
                        p=pack["val"].p[mask_dev] if hasattr(pack["val"], "p") else None
                    ).mean().item()
                else:
                    best_obj_masked = float("nan")

                # x1_norm_t = torch.as_tensor(x1_norm, device=DEVICE, dtype=torch.float32)
                # x2_norm_t = torch.as_tensor(x2_norm, device=DEVICE, dtype=torch.float32)
                # bc_violation = torch.maximum(x1_norm_t, x2_norm_t)

                print("Ipopt (ipopt_box_qp) best objective:", best_obj)
                if ok_mask_full.any():
                    print("Ipopt best objective (learned-converged subset):", best_obj_masked)
                print("Mean Bound constraint violation: ", bc_violation.mean())
                print("Max Bound constraint violation: ", bc_violation.max())
                try:
                    import numpy as np
                    print("Ipopt total time (sum of parallel times):", float(np.sum(para_times)))
                except Exception:
                    print("Ipopt parallel time:", para_times)
                print("Ipopt iterations:", ip_iters)

        #print(f"\nTotal evaluation time over all datasets: {total_time_all:.4f} s")
    elif args.mode == "model_comparison":
        with torch.no_grad():
            import time
            import matplotlib.pyplot as plt
            BATCH = 10000
            if not args.compare_models:
                print("Please provide at least one checkpoint via --compare_models for model_comparison mode.")
                sys.exit(1)

            def _build_curve(times, ok_flags):
                times = np.array(times, dtype=float)
                ok_flags = np.array(ok_flags, dtype=bool)
                times[~ok_flags] = np.inf
                finite = np.isfinite(times)
                if not finite.any():
                    return np.array([0.0, 1.0]), np.array([0.0, 0.0])
                sorted_times = np.sort(times[finite])
                frac = np.arange(1, len(sorted_times) + 1) / len(times)
                sorted_times = np.concatenate([[0.0], sorted_times])
                frac = np.concatenate([[0.0], frac])
                return sorted_times, frac

            for ds_idx, pack in enumerate(pools):
                test_pool = pack["val"]
                N = move_pool_to_device(test_pool, DEVICE)
                N = test_pool.Q.shape[0]
                print(f"\nModel comparison: {mat_names[ds_idx]} | N={N} | batch={BATCH}")

                def _infer_hparams_from_name(path):
                    hs = args.hidden_size
                    nl = args.num_layers
                    m = re.search(r"hiddensize(\d+)", os.path.basename(path))
                    if m:
                        hs = int(m.group(1))
                    m = re.search(r"numlayers(\d+)", os.path.basename(path))
                    if m:
                        nl = int(m.group(1))
                    return hs, nl

                def run_model(ckpt_path, label):
                    hs_use, nl_use = _infer_hparams_from_name(ckpt_path)
                    net_cmp = PS_L20_LSTM(
                        problem=pools[0]["train"],
                        hidden_size=hs_use,
                        num_layers=nl_use,
                        K_inner=K_INNER,
                        device=DEVICE
                    ).to(DEVICE)
                    net_cmp.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                    net_cmp.eval()
                    dummy_opt_cmp = optim.Adam(net_cmp.parameters(), lr=LR_META)
                    conv_times_cmp, conv_ok_cmp = [], []
                    perm_cmp = torch.arange(N, device=test_pool.Q.device)
                    for i in range(0, N, BATCH):
                        idxs = perm_cmp[i:i+BATCH].to(DEVICE).long()
                        mini = slice_pool(test_pool, idxs, N)
                        info_cmp = solve_one_qcqp(
                            dummy_opt_cmp, mini, net_cmp,
                            train_test='test', return_info=True, backprop_every=args.backprop
                        )
                        conv_batch = info_cmp.get("conv_time", None) if isinstance(info_cmp, dict) else None
                        ok_mask = info_cmp.get("ok_mask", None) if isinstance(info_cmp, dict) else None
                        if conv_batch is not None:
                            conv_arr = conv_batch.view(-1).cpu().tolist()
                            ok_list = ok_mask.view(-1).cpu().tolist() if ok_mask is not None else [True] * len(conv_arr)
                            conv_times_cmp.extend(conv_arr)
                            conv_ok_cmp.extend([bool(x) for x in ok_list])
                    t_cmp, frac_cmp = _build_curve(conv_times_cmp, conv_ok_cmp)
                    return label, t_cmp, frac_cmp

                curves = []
                # include base ckpt first
                curves.append(run_model(args.ckpt, os.path.basename(args.ckpt)))
                for ckpt_path in args.compare_models:
                    curves.append(run_model(ckpt_path, os.path.basename(ckpt_path)))

                plot_dir = os.path.join("Plots", "QP_BC")
                os.makedirs(plot_dir, exist_ok=True)
                n_dim = getattr(test_pool, "num_var", None)
                if n_dim is None:
                    n_dim = test_pool.Q.shape[1] if hasattr(test_pool, "Q") else 0
                problem_tag = f"QP_BC_convex_{n_dim}"

                plt.figure()
                colors = ["b", "g", "m", "c", "y", "k"]
                for ci, (lbl, t_curve, f_curve) in enumerate(curves):
                    plt.step(t_curve, f_curve, where="post", label=lbl, color=colors[ci % len(colors)])
                plt.xlabel("Wall time to converge (s)")
                plt.ylabel("Fraction of problems converged")
                plt.title("Convergence vs Time (model comparison)")
                plt.xlim(left=0)
                plt.ylim(0, 1.05)
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.5)
                compare_path = os.path.join(plot_dir, f"{problem_tag}_convergence_compare_models.png")
                plt.savefig(compare_path, bbox_inches="tight")
                print(f"Saved convergence comparison plot to {compare_path}")

    elif args.mode == "profile":
        # Convergence vs time: learned optimizer vs Ipopt
        with torch.no_grad():
            import time
            import matplotlib.pyplot as plt
            net.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
            print(f"✓ loaded weights from {args.ckpt}")
            net.to(DEVICE).eval()

            BATCH = 10000

            conv_times_learned = []
            conv_ok_learned = []
            conv_times_ipopt = []
            conv_ok_ipopt = []
            obj_learned_all = []
            bc_learned_all = []
            obj_ipopt_all = []
            bc_ipopt_all = []
            mean_obj_over_time = []
            mean_obj_timepoints = []
            obj_trace_times = None
            obj_trace_vals = None
            ok_mask_final = None

            for ds_idx, pack in enumerate(pools):
                test_pool = pack["val"]
                N = move_pool_to_device(test_pool, DEVICE)
                N = test_pool.Q.shape[0]
                print(f"\nProfiling convergence: {mat_names[ds_idx]} | N={N} | batch={BATCH}")

                dummy_opt = optim.Adam(net.parameters(), lr=LR_META)
                perm = torch.arange(N, device=test_pool.Q.device)
                num_batches = (N + BATCH - 1) // BATCH

                for i in range(0, N, BATCH):
                    idxs = perm[i:i+BATCH].to(DEVICE).long()
                    mini = slice_pool(test_pool, idxs, N)

                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t0 = time.perf_counter()
                    info = solve_one_qcqp(dummy_opt, mini, net, train_test='test', return_info=True, backprop_every=args.backprop)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t1 = time.perf_counter()

                    batch_time = t1 - t0
                    conv_batch = info.get("conv_time", None) if isinstance(info, dict) else None
                    ok_mask = info.get("ok_mask", None) if isinstance(info, dict) else None
                    if conv_batch is not None:
                        conv_arr = conv_batch.view(-1).cpu().tolist()
                        ok_list = ok_mask.view(-1).cpu().tolist() if ok_mask is not None else [True] * len(conv_arr)
                        conv_times_learned.extend(conv_arr)
                        conv_ok_learned.extend([bool(x) for x in ok_list])
                        if ok_mask is not None:
                            ok_mask_final = ok_mask.detach().cpu()
                        if "obj_per_sample" in info:
                            obj_learned_all.extend(info["obj_per_sample"].view(-1).cpu().tolist())
                        if "bc_violation" in info:
                            bc_learned_all.extend(info["bc_violation"].view(-1).cpu().tolist())
                        if ("obj_traj_times" in info) and ("obj_traj_vals" in info) and info["obj_traj_times"]:
                            obj_trace_times = info["obj_traj_times"]
                            obj_trace_vals = info["obj_traj_vals"]
                        mean_obj_over_time.append(float(info["obj"] if isinstance(info, dict) and "obj" in info else np.nan))
                        mean_obj_timepoints.append(conv_arr[0] if conv_arr else batch_time)
                    else:
                        conv_times_learned.extend([batch_time] * len(idxs))
                        if ok_mask is not None:
                            conv_ok_learned.extend([bool(x) for x in ok_mask.view(-1).tolist()])
                            ok_mask_final = ok_mask.detach().cpu()
                        else:
                            conv_ok_learned.extend([True] * len(idxs))
                        if "obj_per_sample" in info:
                            obj_learned_all.extend(info["obj_per_sample"].view(-1).cpu().tolist())
                        if "bc_violation" in info:
                            bc_learned_all.extend(info["bc_violation"].view(-1).cpu().tolist())
                        if ("obj_traj_times" in info) and ("obj_traj_vals" in info) and info["obj_traj_times"]:
                            obj_trace_times = info["obj_traj_times"]
                            obj_trace_vals = info["obj_traj_vals"]
                        mean_obj_over_time.append(float(info["obj"] if isinstance(info, dict) and "obj" in info else np.nan))
                        mean_obj_timepoints.append(batch_time)
                    print(f"  learned batch time: {batch_time:.4f}s")

                # Ipopt per-problem solves
                sols, _, para_times, ip_iters, bc_violation = pack["val"].opt_solve(
                    solver_type='ipopt_box_qp_extended',
                    tol=1e-2
                )
                if isinstance(bc_violation, torch.Tensor):
                    bc_violation_list = bc_violation.detach().cpu().view(-1).tolist()
                else:
                    bc_violation_list = list(bc_violation) if bc_violation is not None else []
                if para_times is None:
                    para_times = []
                if isinstance(para_times, (float, int)):
                    para_times = [float(para_times) / max(1, N)] * N
                elif hasattr(para_times, "flatten"):
                    para_times = para_times.flatten().tolist()
                else:
                    para_times = list(para_times)
                # Ipopt runs sequentially; convert per-problem durations to completion times
                cum_times = np.cumsum(para_times).tolist() if para_times else []
                for idx_t, t_ip in enumerate(cum_times):
                    conv_times_ipopt.append(t_ip)
                    conv_ok_ipopt.append(True)
                    if idx_t < len(sols):
                        # shape inputs for single-sample obj_fn
                        x_single = torch.tensor(sols[idx_t], device=DEVICE, dtype=torch.float32).view(1, -1, 1)
                        Q_single = pack["val"].Q[idx_t:idx_t+1]
                        p_single = pack["val"].p[idx_t:idx_t+1] if hasattr(pack["val"], "p") else None
                        obj_val = pack["val"].obj_fn(x_single, Q=Q_single, p=p_single).item()
                        obj_ipopt_all.append(float(obj_val))
                        bc_ipopt_all.append(float(bc_violation_list[idx_t]) if idx_t < len(bc_violation_list) else float("nan"))

            def _build_curve(times, ok_flags):
                times = np.array(times, dtype=float)
                ok_flags = np.array(ok_flags, dtype=bool)
                times[~ok_flags] = np.inf
                finite = np.isfinite(times)
                if not finite.any():
                    return np.array([0.0, 1.0]), np.array([0.0, 0.0])
                sorted_times = np.sort(times[finite])
                frac = np.arange(1, len(sorted_times) + 1) / len(times)
                sorted_times = np.concatenate([[0.0], sorted_times])
                frac = np.concatenate([[0.0], frac])
                return sorted_times, frac

            t_learned, frac_learned = _build_curve(conv_times_learned, conv_ok_learned)
            t_ipopt, frac_ipopt = _build_curve(conv_times_ipopt, conv_ok_ipopt)

            plot_dir = os.path.join("Plots", "QP_BC")
            os.makedirs(plot_dir, exist_ok=True)
            n_dim = getattr(test_pool, "num_var", None)
            if n_dim is None:
                # fallback to inferred shape if available
                n_dim = test_pool.Q.shape[1] if hasattr(test_pool, "Q") else 0
            problem_tag = f"QP_BC_convex_{n_dim}"

            plt.figure()
            plt.step(t_learned, frac_learned, where="post", label="learned", color="b")
            plt.step(t_ipopt, frac_ipopt, where="post", label="ipopt", color="r")
            plt.xlabel("Wall time to converge (s)")
            plt.ylabel("Fraction of problems converged")
            plt.title("Convergence vs Time (validation)")
            plt.xlim(left=0)
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            conv_path = os.path.join(plot_dir, f"{problem_tag}_convergence_vs_time.png")
            plt.savefig(conv_path, bbox_inches="tight")
            print(f"Saved convergence vs time plot to {conv_path}")

            # Objective vs time using ONLY the problems that eventually converged (fixed subset)
            obj_plot_x = []
            obj_plot_y = []
            if obj_trace_times and obj_trace_vals and ok_mask_final is not None and ok_mask_final.any():
                t0 = obj_trace_times[0]
                for t, v in zip(obj_trace_times, obj_trace_vals):
                    obj_plot_x.append(t - t0)
                    obj_plot_y.append(float(v[ok_mask_final].mean()))
            ipopt_final_mean = float(np.mean([v for v, ok in zip(obj_ipopt_all, conv_ok_learned) if ok])) if obj_ipopt_all else float("nan")

            plt.figure()
            if obj_plot_x and obj_plot_y:
                plt.plot(obj_plot_x, obj_plot_y, label="learned mean objective (converged)", color="b")
                plt.scatter(obj_plot_x, obj_plot_y, s=12, color="b", alpha=0.7)
            if not np.isnan(ipopt_final_mean):
                plt.axhline(ipopt_final_mean, color="r", linestyle="--", label="ipopt mean (learned-converged subset)")
            plt.xlabel("Wall time (s)")
            plt.ylabel("Objective value")
            plt.title("Objective vs Time (validation)")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            obj_gap_path = os.path.join(plot_dir, f"{problem_tag}_objective_gap_vs_time.png")
            plt.savefig(obj_gap_path, bbox_inches="tight")
            print(f"Saved objective gap vs time plot to {obj_gap_path}")

            # Bound constraint violation vs time (L2 norm)
            plt.figure()
            plt.scatter(conv_times_learned, bc_learned_all, s=10, alpha=0.6, label="learned", color="b")
            plt.scatter(conv_times_ipopt, bc_ipopt_all, s=10, alpha=0.6, label="ipopt", color="r")
            plt.xlabel("Wall time to converge (s)")
            plt.ylabel("Bound constraint violation (L2)")
            plt.yscale("log")
            plt.title("Bound Constraint Violation vs Time (validation)")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            bc_path = os.path.join(plot_dir, f"{problem_tag}_bc_violation_vs_time.png")
            plt.savefig(bc_path, bbox_inches="tight")
            print(f"Saved bound constraint violation vs time plot to {bc_path}")

        
