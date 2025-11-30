import argparse, time, torch, torch.optim as optim
import torch.nn.init as init

import os
from problems.Convex_QCQP import Convex_QCQP        
from problems.QP_extended import QP
from models.LSTM_L20_Projected_Math_extended import PS_L20_LSTM         

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

#DIM_X    = 10        # decision variables
DIM_S    = 50         # inequality slacks
HIDDEN   = 128        # LSTM hidden size, did hidden size 32 make the model better? can play around with this
K_INNER  = 10         # micro-steps per outer iteration, need to fix this in the code

EPOCHS   = 10                    # meta-training epochs
LR_META  = 1e-3                 # learning-rate for Adam

MAX_OUTER  = 1000
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
                   max_outer=MAX_OUTER, train_test = 'train', val_stop = 5, problem_type = 'portfolio'):

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

    eps_t = torch.tensor(1e-10, device=device)

    lb_vals = problem.lb.expand(B, n).to(device)   
    ub_vals = problem.ub.expand(B, n).to(device)   

    lb = lb_vals.unsqueeze(-1)
    ub = ub_vals.unsqueeze(-1)

    mask_lb = torch.isfinite(lb_vals)  
    mask_ub = torch.isfinite(ub_vals)  

    # create branch for type of initialization based on the type of problem and its purpose
    if problem_type == 'normal':
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
            # if problem.num_eq > 0:
            #     A_pinv = torch.linalg.pinv(problem.A) 
            #     x  = torch.bmm(A_pinv, problem.b).to(device)
            # else:
            #     x = x_init.unsqueeze(-1)
            xE = x.clone()

            # initialize the duals anbd the shifts            
            muB = x.new_tensor([muB0], device=device)
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
            x  = torch.zeros(B, n, 1, device=device)
            v = None
            vE = None
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
    
    #chi_max = torch.minimum(chi, torch.tensor(1000.0, device=chi.device, dtype=chi.dtype))

    
    
    if device == 'cpu':
        # display merit/residual
        M_k = problem.merit_M(x, s,      
                y, v, z1, z2, w1, w2,    
                x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
                muP, muB, muA, bad_x1, bad_x2).mean()
        chi = problem.chi(x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA)
        print(f"[IPM‐iter {0}] μP={muP.mean().item():.1e}, "
                f"μB={muB.mean().item():.1e}, μA={muA.mean().item():.1e}, M={M_k:.4e}, ∥r∥={chi_max.mean().item():.4e}")
        
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
 
    if device == 'cpu':
        P = problem.primal_feasibility(x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA)      
        # dual feasibility
        D = problem.dual_feasibility(x, s,      
                y, v, z1, z2, w1, w2,    
                x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
                muP, muB, muA)        
        # complementarity
        C = problem.complementarity(x, s,      
                y, v, z1, z2, w1, w2,    
                x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
                muP, muB, muA)   
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

    

    # outer IPM iterations
    for outer in range(1, max_outer+1):

        
        # flatten for LSTM call
        x = x.squeeze(-1)   
        x1 = x1.squeeze(-1)   
        x2= x2.squeeze(-1)
        z1 = z1.squeeze(-1)
        z2 = z2.squeeze(-1)
        if problem.num_ineq > 0:
            s = s.squeeze(-1)
            y = y.squeeze(-1)
            w1 = w1.squeeze(-1)
        if problem.num_eq > 0:
            v = v.squeeze(-1)

        N = 7 * n

        # currently does 1 micro-iteration of the LSTM, can change that 
        (x, s, y, v, z1, z2, w1, w2), step_loss_vector, (lstm_states_x, lstm_states_s, r_k) = net(
            problem,
            x, s, y, v, z1, z2, w1, w2, 
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA, r_k,
            lstm_states_x, lstm_states_s, bad_x1, bad_x2, M_max, n_vec = n_vec, 
            project_step=project_tube_all, outer = outer
        )


        #print(step_loss_vector.shape)
        step_loss = step_loss_vector.mean()

        step_loss = step_loss/K_INNER


        if device == 'cpu':
            print("Loss for this step: ", step_loss)

        total_inner_loss = total_inner_loss + (step_loss)

        P = problem.primal_feasibility(x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA) 

        D = problem.dual_feasibility(x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA) 
        C = problem.complementarity(x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA) 
        
        total_inner_loss = total_inner_loss + (P + D + C).mean()

        total_inner_loss = total_inner_loss/n

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


        


        # meta_opt.zero_grad(set_to_none=True)      # clear old gradients
        # step_loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        # #torch.autograd.set_detect_anomaly(True)
        # meta_opt.step()                           # update the LSTM weights

        #torch.autograd.set_detect_anomaly(True)
        if train_test == 'train':
            #if outer % 10 == 0:
            #torch.autograd.set_detect_anomaly(True)
            meta_opt.zero_grad(set_to_none=True)      # clear old gradients
            #total_inner_loss = total_inner_loss/20
            total_inner_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            #torch.autograd.set_detect_anomaly(True)
            meta_opt.step()                           # update the LSTM weights

            total_inner_loss = 0

            h_x, c_x = lstm_states_x
            h_s, c_s = lstm_states_s
            lstm_states_x = (h_x.detach(), c_x.detach())
            if problem.num_ineq > 0:
                lstm_states_s = (h_s.detach(), c_s.detach())
            r_k = r_k.detach()

            x = x.detach()
            z1 = z1.detach()
            z2 = z2.detach()
            if problem.num_eq > 0:
                v = v.detach()
            if problem.num_ineq > 0:
                y = y.detach()
                w1 = w1.detach()
                s = s.detach()

            x1E = x1E.detach(); x2E = x2E.detach();z1E = z1E.detach();z2E = z2E.detach()
            if problem.num_ineq > 0:
                s1E = s1E.detach(); yE = yE.detach(); w1E = w1E.detach()

            if problem.num_eq > 0:
                vE = vE.detach()
            muB = muB.detach(); muP = muP.detach(); muA = muA.detach()

            # 
        
        chi = P + D + C

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

            
            obj_fcn = problem.obj_fn(x).mean().item()

            chi = problem.chi(x, s,      
                y, v, z1, z2, w1, w2,    
                x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
                muP, muB, muA)
            print(f"[IPM‐iter {outer}] μP={muP.mean().item():.1e}, "
                f"μB={muB.mean().item():.1e}, μA={muA.mean().item():.1e}, M={M_k_new:.4e}, ∥r∥={chi.mean().item():.4e}")
            
            print("x: ", x.mean().item())
            print("z1: ", z1.mean().item())
            print("z2: ", z2.mean().item())

            for i in range(min(5, lb.shape[0])):
                print(f"obj[{i}] = {problem.obj_fn(x).view(B)[i]:.6f}")
       
  
        if device == 'cpu':
            print("Primal Feasability: ", P.mean())
            for i in range(min(5, lb.shape[0])):
                print(f"Primal Feasability[{i}] = {P.view(B)[i]:.6f}")
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
            print("CONVERGED ")
        thresh = 1e-2
        ok = (P < thresh) & (D < thresh) & (C < thresh)   # elementwise
        print("Percentage converged: ", sum(ok)/B)


        # early stopping mechanism
        if train_test != "train":
            thresh = 1e-2
            ok = (P < thresh) & (D < thresh) & (C < thresh)   # elementwise
            if ok.all().item():   # True iff ALL problems pass
                return outer 

        mask_lb_3d = mask_lb.unsqueeze(-1) if mask_lb.dim() == 2 else mask_lb  # [B,n,1]
        mask_ub_3d = mask_ub.unsqueeze(-1) if mask_ub.dim() == 2 else mask_ub

        raw_x1 = x - lb_vals.unsqueeze(-1)        # [B,n,1]
        x1     = torch.where(mask_lb_3d, raw_x1, torch.zeros_like(raw_x1))

        raw_x2 = ub_vals.unsqueeze(-1) - x        # [B,n,1]
        x2     = torch.where(mask_ub_3d, raw_x2, torch.zeros_like(raw_x2))
        
        if problem.num_ineq > 0:
            s1 = s.clone()

        Mtest = problem.Mtest(x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA, M_max, bad_x1, bad_x2)   
        
        # print("Variables passing Mtest: ", (Mtest.abs() < 0.1).sum(dim = 0))

        # if (Mtest.abs() < 0.1).all():
        #     print("SUCCESS")
        #     return True 

        #if (M_k_new < (0.8*M_k)):
        

        mask_O = chi < chi_max               # [B]  O-iterate candidates

        # print(chi[0])
        # print(chi_max[0])
        
        mask_M = (~mask_O) & (Mtest < tolM)  # M if not O

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
                
                # 1) Primal check
                bad_primal = P[mask_M] > tolM_M.squeeze(-1).squeeze(-1)           # [B_M]

                muP_new[mask_M] = torch.where(bad_primal,
                                            muP[mask_M]/2,
                                            muP[mask_M])
            


            

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
            vE.copy_(vE_new)


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

        M_k = M_k_new
    print("Loss for this step: ", step_loss_vector.mean())
    print("Primal Feasability: ", P.mean())
    print("Dual Feasability: ", D.mean()) # stationarity is giving me issues
    print("Complementarity: ", C.mean())
    thresh = 1e-2
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

    mat_names = [
        "portfolio_QP_box_s10_t2_eq3"
    ]

    #mat_names = ["QP_convex_10var_3eq_3ineq_bounded"]

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

        BATCH = 25
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

                    _ = solve_one_qcqp(meta_opt, mini, net)
                    # get the objective 
                    # evaluate(val_pool, net, tag="val")
                print(f"Epoch {ep:3d} | {time.time()-t0:.1f}s")
                # if flag:
                #     break
            #solve_one_qcqp(meta_opt, pools[0]['val'], net, train_test = 'val')
        torch.save(net.state_dict(), args.ckpt)
        print(f"✓ weights saved to {args.ckpt}")

    # test evaluation 
    else:
        import time
        net.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
        print(f"✓ loaded weights from {args.ckpt}")
        net.to(DEVICE).eval()

        import copy
        import numpy as np

        BATCH = 1  # set your eval batch size
        total_time_all = 0.0  # total time over all datasets

        for ds_idx, pack in enumerate(pools):
            test_pool = pack["val"]   # or pack["test"] if you want to eval on test
            N = test_pool.Q.shape[0]
            print(f"\nTesting: {mat_names[ds_idx]} | N={N} | batch={BATCH}")

            # a dummy optimizer if solve_one_qcqp expects one (won't be stepped in 'test' mode)
            dummy_opt = optim.Adam(net.parameters(), lr=LR_META)

            perm = torch.arange(N, device=test_pool.Q.device)
            num_batches = (N + BATCH - 1) // BATCH

            iterations = []
            ds_start = time.perf_counter()  # start timing this dataset

            for i in range(0, N, BATCH):
                b = i // BATCH + 1
                print(f"BATCH: {b}/{num_batches}")

                if DEVICE == 'cpu':
                    idxs = perm[i:i+BATCH]            # LongTensor on CPU
                    idxs_list = idxs.tolist()

                    mini = copy.copy(test_pool)
                    for name, val in test_pool.__dict__.items():
                        if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.size(0) == N:
                            setattr(mini, name, val[idxs])
                        elif isinstance(val, np.ndarray) and val.shape and val.shape[0] == N:
                            setattr(mini, name, val[idxs.numpy()])
                        elif isinstance(val, list) and len(val) == N:
                            setattr(mini, name, [val[j] for j in idxs_list])
                else:
                    idxs = perm[i:i+BATCH].to(DEVICE).long()
                    idxs_list = idxs.cpu().tolist()

                    mini = copy.copy(test_pool)
                    for name, val in test_pool.__dict__.items():
                        if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.size(0) == N:
                            setattr(mini, name, val[idxs])
                        elif isinstance(val, np.ndarray) and val.shape and val.shape[0] == N:
                            tval = torch.from_numpy(val).to(DEVICE)
                            setattr(mini, name, tval[idxs])
                        elif isinstance(val, list) and len(val) == N:
                            setattr(mini, name, [val[j] for j in idxs_list])

                # run one batched solve in test mode (no learning updates inside)
                batch_start = time.perf_counter()
                iters = solve_one_qcqp(dummy_opt, mini, net, train_test='test')
                batch_end = time.perf_counter()

                iterations.append(iters)
                print(f"  batch time: {batch_end - batch_start:.4f} s, iterations: {iters}")

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
            sols, _, para_times, ip_iters = pack["val"].opt_solve(
                solver_type='ipopt_box_qp_extended',
                tol=1e-2
            )

            # Evaluate objective of the Ipopt solutions
            sols_t = torch.tensor(sols).unsqueeze(-1).float().to(DEVICE)
            best_obj = pack["val"].obj_fn(sols_t).mean().item()

            print("Ipopt (ipopt_box_qp) best objective:", best_obj)
            print("Ipopt parallel time:", para_times)
            print("Ipopt iterations:", ip_iters)

        #print(f"\nTotal evaluation time over all datasets: {total_time_all:.4f} s")

              
                

    #evaluate(test_pool, net, tag="test")
