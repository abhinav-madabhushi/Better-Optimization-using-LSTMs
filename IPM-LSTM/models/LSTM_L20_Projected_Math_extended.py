import torch, torch.nn as nn, torch.nn.functional as F

class PS_L20_LSTM(nn.Module):
    # ------------------------------------------------------------
    def __init__(self, problem, hidden_size=64, num_layers=2, K_inner = 1, device='cuda' if torch.cuda.is_available() else 'cpu', fixedmuNumber = 15): 
        super().__init__()
        self.K_inner = K_inner

        self.fixedmuNumber = fixedmuNumber

        self.device = device

        self.px_feat_dim = 19
        self.lstm_px = nn.LSTM(
            input_size  = self.px_feat_dim, #+ (2*hidden_size),   # independent of n
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
        )
        self.px_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 2)   # per-token scalar step
        )

        self.ps_feat_dim = 11
        self.lstm_ps = nn.LSTM(
            input_size  = self.ps_feat_dim, #+ (2*hidden_size),   # independent of n
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
        )
        self.ps_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)   # per-token scalar step
        )


    # # ------------------------------------------------------------
    # def _lstm_cell(self, x_in, h_prev, c_prev):
    #     i = torch.sigmoid(x_in @ self.W_i + h_prev @ self.U_i + self.b_i)
    #     f = torch.sigmoid(x_in @ self.W_f + h_prev @ self.U_f + self.b_f)
    #     o = torch.sigmoid(x_in @ self.W_o + h_prev @ self.U_o + self.b_o)
    #     u = torch.tanh(   x_in @ self.W_u + h_prev @ self.U_u + self.b_u)
    #     c = i*u + f*c_prev
    #     h = o * torch.tanh(c)
    #     return h, c

    # inside class PS_L20_LSTM(nn.Module):

    def forward(self,
            problem,
            x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA,
            r_k,                                  # history‐encoder [B, N+2]
            states_x, states_s, bad_x1, bad_x2,  M_max, n_vec,                           # (h_prev, c_prev) for nn.LSTM
            project_step, outer):
        
        # add for loop here to do multiple iterations of the forward pass before backpropogating 

        # define the sizes
        B, n     = x.shape
        m_ineq   = problem.num_ineq
        meq     = problem.num_eq 

        # convert to column form for merit grad function
        # reshaping the data
        # x = x.unsqueeze(-1)
        # x1 = x1.unsqueeze(-1)
        # x2 = x2.unsqueeze(-1)
        # z1 = z1.unsqueeze(-1)
        # z2 = z2.unsqueeze(-1)
        # x = x.unsqueeze(-1)
        # s = s.unsqueeze(-1)
        # y = y.unsqueeze(-1)
        # v = v.unsqueeze(-1)
        # w = w.unsqueeze(-1)
        #print("x shape at start: ", x.shape)
        
        total = 0

        step_losses = []

        for i in range (1, self.K_inner + 1):
            def _trim_grad(g, width):
                if g is None:
                    return None
                g = g.reshape(B, -1)
                if g.shape[1] > width:
                    g = g[:, :width]
                elif g.shape[1] < width:
                    pad = g.new_zeros(B, width - g.shape[1])
                    g = torch.cat([g, pad], dim=1)
                return g

            # slack1 = x1.squeeze(-1) + muB      # should stay > 0
            # slack2 = x2.squeeze(-1) + muB      # should stay > 0

            # # 2) build a mask of infeasible entries
            # bad_x1 = (slack1 <= 0)   # shape [B,n,1], True wherever x1+μB ≤ 0
            # bad_x2 = (slack2 <= 0)   # shape [B,n,1], True wherever x2+μB ≤ 0

            # total_infeas = int(bad_x2.sum().item())
            # #print(f"Total infeasible variables wrt to x2 in batch: {total_infeas}")

            # slackz1 = (z1.squeeze(-1) + muB)    # [B,n]
            # slackz2 = (z2.squeeze(-1) + muB)    # [B,n]

            # # build masks of the exact indices that went non-positive
            # bad_z1 = (slackz1 <= 0).unsqueeze(-1)  # [B,n,1]
            # bad_z2 = (slackz2 <= 0).unsqueeze(-1)  # [B,n,1]

            # # # Option A: out-of-place with torch.where
            # # z1 = torch.where(bad_z1, torch.zeros_like(z1), z1)
            # # z2 = torch.where(bad_z2, torch.zeros_like(z2), z2)

            # obtaining each of the gradients
            grad_x, grad_s, grad_y, grad_vv, grad_z1, grad_z2, grad_w1 = problem.merit_grad_M(
                x, s,      
                y, v, z1, z2, w1, w2,    
                x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
                muP, muB, muA, bad_x1, bad_x2
                )
        
            # flatten gradients
            grad_x  = _trim_grad(grad_x, n)
            grad_z1 = _trim_grad(grad_z1, n)
            grad_z2 = _trim_grad(grad_z2, n)
            grad_s  = _trim_grad(grad_s, m_ineq)
            grad_y  = _trim_grad(grad_y, m_ineq)
            grad_vv = _trim_grad(grad_vv, meq)
            grad_w1 = _trim_grad(grad_w1, m_ineq)



            #m = s.shape[1]

            # if n_vec is not None:
            #     # per-sample true dimensions
            #     base = (torch.arange(n, device=self.device)[None, :] < n_vec[:, None]).float()  # [B,n]
            #     mask_x = base
            #     mask_z1 = base
            #     mask_z2 = base
            #     mask_v = torch.cat([mask_x, mask_z1, mask_z2], dim=1)                           # [B,m]
            #     mask_H = mask_v.unsqueeze(2) * mask_v.unsqueeze(1)                              # [B,m,m]
            # else:
            #     # fixed dimension → just ones everywhere
            #     # fixed dimension → just ones everywhere
            #     mask_x = torch.ones(B, n, device=self.device)
            #     mask_z1 = mask_x
            #     mask_z2 = mask_x
            #     mask_s = torch.ones(B, m, device=self.device)
            #     mask_y = mask_s
            #     mask_vv = mask_x
            #     mask_w1 = mask_s
            #     mask_v = torch.ones(B, N, device=self.device)
            #     mask_H = torch.ones(B, N, N, device=self.device)

            if self.device == 'cpu':
                print("x: ", x.mean().item())
                # print("x1: ", x1.mean().item())
                # print("x2: ", x2.mean().item())
                print("z1: ", z1.mean().item())
                print("z2: ", z2.mean().item())
                if problem.num_ineq > 0:
                    print("y: ", y.mean().item())
                    print("w1: ", w1.mean().item())
                    print("s: ", s.mean().item())
                if problem.num_eq > 0:
                    print("v: ", v.mean().item())

                # print("x: ", x)
                # print("x1: ", x1)
                # print("x2: ", x2)
                # print("z1: ", z1)
                # print("z2: ", z2)

            # mask_lb = torch.isfinite(problem.lb)   # [B,n,1], True where ℓ_j > -∞
            # mask_ub = torch.isfinite(problem.ub)   # [B,n,1], True where u_j < +∞

            eps = 1e-12

            if torch.is_tensor(muB):
                muBv = muB.view(B, 1)
            else:
                muBv = torch.full((B, 1), float(muB), device=self.device)
            
            # bounds
            lb = problem.lb.expand(B, n)                             # [B,n]
            ub = problem.ub.expand(B, n)                             # [B,n]
            has_lb = torch.isfinite(lb)                             # [B,n]
            has_ub = torch.isfinite(ub)                              # [B,n]

            # slacks for box constraints
            x1 = torch.where(has_lb, x - lb, torch.zeros_like(x))  # [B,n]
            x2 = torch.where(has_ub, ub - x, torch.zeros_like(x))  # [B,n]

            # ---------- shapes & helpers ----------
            eps = 1e-12
            inv_cap = 1e20
            alpha = 1e-4

            to2d = lambda t: (t.squeeze(-1) if (t is not None and t.dim() == 3) else t)
            B = x.shape[0]

            x1 = to2d(x1);  x2 = to2d(x2)
            z1 = to2d(z1);  z2 = to2d(z2)
            s  = to2d(s)
            w1 = to2d(w1)
            v = to2d(v) if v is not None else None

            # broadcasts for μ
            muB_bn = muB.view(B, 1).expand_as(x1)                                  # [B,n]
            muP_m = (muP.view(B, 1).expand_as(s)    if s  is not None else None)  # [B,m]
            muP_v = (muP.view(B, 1).expand_as(v)    if v  is not None else None)  # [B,m]

            muP_x = (muP.view(B, 1).expand_as(x)    if v  is not None else None)  # [B,m]

            muA_p = (muA.view(B, 1).expand(B, v.shape[1]) if v is not None else None)

            # ---------- Z-block diagonals (box): D^z and its inverse ----------
            # D^z_1 = (x1 + μB) / (z1 + μB),  invD^z_1 = (z1 + μB) / (x1 + μB)
            invD1 = (z1 + muB_bn) / (x1 + muB_bn + eps)        # [B,n]
            invD2 = (z2 + muB_bn) / (x2 + muB_bn + eps)        # [B,n]
            invD1 = invD1.clamp_min(0.0).clamp_max(inv_cap)
            invD2 = invD2.clamp_min(0.0).clamp_max(inv_cap)

            # ---------- W-block diagonals (ineq slacks): D^w and its inverse ----------
            if s is not None:
                # lower-bound on s (defaults to 0 if not provided)
                s_lb = getattr(problem, 's_lb', torch.zeros_like(s))       # [B,m]
                mask_s_lb = torch.isfinite(s_lb)                            # [B,m]
                s1_ = torch.where(mask_s_lb, s - s_lb, torch.zeros_like(s))
                muB_m = muBv.view(B, 1).expand_as(s1_)                      # [B,m]

                # D^w_1 = (s1 + μB) / (w1 + μB),  invD^w_1 = (w1 + μB) / (s1 + μB)
                if w1 is not None:
                    Dw1_inv = (w1 + muB_m) / (s + muB_m + eps)           # [B,m]
                    Dw1_inv = torch.where(mask_s_lb, Dw1_inv, torch.zeros_like(Dw1_inv))
                    Dw1_inv = Dw1_inv.clamp_min(0.0).clamp_max(inv_cap)
                    # Dw1 = (1.0 / (Dw1_inv + eps)).clamp_min(0.0).clamp_max(inv_cap)
                    Dw1 = (s + muB_m) / (w1 + muB_m + eps) 
                    Dw1 = Dw1.clamp_min(0.0).clamp_max(inv_cap)
                else:
                    Dw1_inv = Dw1 = None

            else:
                Dw1_inv = Dw1 = Dw2_inv = Dw2 = None
                mask_s_lb = None

            # ---------- Simple diagonal blocks for multipliers ----------
            # D_Y = μ^P I  and  D_A = μ^A I
            if muP_m is not None:
                DY  = muP_m.clamp_min(eps)                          # [B,m]
                DY_inv = (1.0 / DY).clamp_min(0.0).clamp_max(inv_cap)
            else:
                DY = DY_inv = None
            
            if muP_v is not None:
                DV  = muP_v.clamp_min(eps)                          # [B,m]
                DV_inv = (1.0 / DV).clamp_min(0.0).clamp_max(inv_cap)
            else:
                DV = DV_inv = None

            # if muA_p is not None:
            #     DA  = muA_p.clamp_min(eps)                          # [B,meq]
            #     DA_inv = (1.0 / DA).clamp_min(0.0).clamp_max(inv_cap)
            # else:
            #     DA = DA_inv = None
            
            if s is not None and w1 is not None:
                DW_combo = Dw1                                   # [B,m] = (w1+μB)/(s1+μB)
                DW_combo_inv = Dw1                                   # [B,m]
            else:
                DW_combo = DW_combo_inv = None
            
            x1E = to2d(x1E)
            x2E = to2d(x2E)
            z1E = to2d(z1E)
            z2E = to2d(z2E)
            s1E = to2d(s1E)
            w1E = to2d(w1E)
            yE = to2d(yE)
            vE = to2d(vE)

            if muB_bn is not None:
                log_muB_bn = torch.log(muB_bn.clamp_min(eps))   # same shape as muB_bn
            
            if v is not None:
                Atv = torch.bmm(problem.At, v.unsqueeze(-1)).squeeze(-1)
                Atgradvv = torch.bmm(problem.At, grad_vv.unsqueeze(-1)).squeeze(-1)
                AtvE = torch.bmm(problem.At, vE.unsqueeze(-1)).squeeze(-1)
                log_muP_x  = torch.log(muP_x.clamp_min(eps))
            else:
                Atv = torch.zeros_like(x)
                Atgradvv = torch.zeros_like(x)
                AtvE = torch.zeros_like(x)
                log_muP_x  = torch.zeros_like(x)


            #print(problem.G.shape)
            # print(problem.At.shape)
            # Atv = torch.bmm(problem.At, v.unsqueeze(-1))
            # print(Atv.shape)

            # choose the features you want per coordinate (order must match px_feat_dim)
            feat_px = torch.stack([
                x, 
                x1, 
                x2, 
                z1, 
                z2, 
                Atv, 
                grad_x,
                grad_z1,
                grad_z2,
                Atgradvv, 
                x1E,
                x2E,
                z1E,
                z2E,
                AtvE,
                # muB_bn,
                # muP_x,
                log_muB_bn, 
                log_muP_x, 
                invD1, 
                invD2
                #torch.bmm(problem.At, muP_v.unsqueeze(-1)).squeeze(-1),
            ], dim=-1) 

            BN   = B * n
            inp  = feat_px.reshape(BN, self.px_feat_dim)        # [BN, F]
            inp  = inp.unsqueeze(1)



            # if n_vec is not None:
            #     mask = (torch.arange(n, device=x.device)[None,:] < n_vec[:,None]).float()
            #     Y, (h_new, c_new) = self.lstm_px(inp)  
            #     Y = Y * mask.unsqueeze(-1)  
            # else:
            #     Y, (h_new, c_new) = self.lstm_px(inp)          # [B, n, H]
            
            Y, (h_new_x, c_new_x) = self.lstm_px(inp, states_x) 

            # p_x = self.px_head(Y).reshape(B, n)

            p = self.px_head(Y)

            p_x_1, p_x_2 = p.chunk(2, dim=-1)

            # p_x_1 = p_x_1.reshape(B, n)
            # p_x_2 = p_x_2.reshape(B, n)

            p_x_1 = torch.abs(p_x_1.reshape(B, n))
            p_x_2 = torch.abs(p_x_2.reshape(B, n))

            # p_x = p_x.reshape(B, n)
            # p_z1 = p_z1.reshape(B, n)
            # p_z2 = p_z2.reshape(B, n)

            # p_x = F.softplus(p_x)
            # p_x = torch.abs(p_x)  
            #p_x = torch.sigmoid(p_x)  
            # p_z1 = F.softplus(p_z1)   
            # p_z2 = F.softplus(p_z2)    
          

            # if padded, mask p_x too
            # if n_vec is not None:
            #     p_x = p_x * mask
            # print(muP_m)


            if s is not None:
                log_muP_m  = torch.log(muP_m.clamp_min(eps))  
                feat_ps = torch.stack([
                    s, 
                    y, 
                    w1, 
                    grad_s, 
                    grad_y, 
                    grad_w1, 
                    s1E, 
                    yE, 
                    w1E, 
                    log_muP_m, 
                    DW_combo
                ], dim=-1) 

                Bm   = B * m_ineq
                inp  = feat_ps.reshape(Bm, self.ps_feat_dim)        # [BN, F]
                inp  = inp.unsqueeze(1)

                Y, (h_new_s, c_new_s) = self.lstm_ps(inp, states_s) 

                p_s = self.ps_head(Y).reshape(B, m_ineq)

                # p_s, p_y = p.chunk(2, dim=-1)

                # p_s = p_s.reshape(B, m_ineq)
                # p_y = p_y.reshape(B, m_ineq)

                # p_y = torch.abs(p_y)

                p_s = torch.abs(p_s)
                #p_s = torch.sigmoid(p_s)

                #(h_new_s, c_new_s) = states_s
            else:
                (h_new_s, c_new_s) = (None, None)


            # for inequalities (if present)
            if m_ineq > 0:
                # broadcasts of μ^B, μ^P across m_ineq
                muB_m = muB.view(B, 1).expand(B, m_ineq)            # [B,m]
                # s lower bound (defaults to 0); mask for active lower bound
                s_lb = getattr(problem, 's_lb', torch.zeros_like(s)) # [B,m]
                mask_s_lb = torch.isfinite(s_lb)                     # [B,m]
                s1 = torch.where(mask_s_lb, s - s_lb, torch.zeros_like(s))  # [B,m]
                S1mu = (s1 + muB_m).clamp_min(eps)                   # [B,m]
                W1mu = (w1 + muB_m).clamp_min(eps)                   # [B,m]
                # π^W = μ^B (S1^μ)^{-1} (w1^E − s1 + s1^E)
                piW = muB_m * (w1E - s1 + s1E) / S1mu                # [B,m]

                # Jacobian J_f = G for c(x)=Gx−c
                G = problem.G if hasattr(problem, 'G') else self.G   # [B,m,n] or [m,n]
                Gb = G if G.dim() == 3 else G.unsqueeze(0).expand(B, -1, -1)  # [B,m,n]

                denom = DY + DW_combo + eps               # [B, m]
                r = grad_y / denom                        # [B, m]
                Jt_r = torch.einsum('bmn,bm->bn', Gb, r)  # [B, n]  = J^T * r

            # for equalities (if present)
            if meq and meq > 0:
                A = problem.A if hasattr(problem, 'A') else self.A            # [B,meq,n] or [meq,n]
                Ab = A if A.dim() == 3 else A.unsqueeze(0).expand(B, -1, -1)  # [B,meq,n]
                b = problem.b if hasattr(problem, 'b') else self.b            # [B,meq] or [meq]
                #print(b.shape)
                bB = b if b.dim() == 2 else b.squeeze(-1)  # [B,meq]

            # Δx  (given px and grad_x are [B,n])
            #print(Gb.shape)

            # print(DV_inv.shape)
            # print(grad_vv.shape)
            dx = (- p_x_1 * (grad_x)) 
            if s is not None:
                dx = dx + (- p_x_1 * Jt_r)
            if v is not None:
                dx = dx + (- p_x_1 * torch.bmm(problem.At, (DV_inv * grad_vv).unsqueeze(-1)).squeeze(-1))                                     

            # Δz1, Δz2  (Sec. 7 explicit formulas; use x1,x2 and Δx)
            dx1 = torch.where(has_lb,  dx,  torch.zeros_like(dx))            # [B,n]
            dx2 = torch.where(has_ub, -dx,  torch.zeros_like(dx))            # [B,n]
            X1mu = (x1 + muB_bn).clamp_min(eps)                              # [B,n]
            X2mu = (x2 + muB_bn).clamp_min(eps)                              # [B,n]

            dz1 = -( z1 * (x1 + dx + muB_bn) - (muB_bn * z1E) + muB_bn * (- x1E + x1 + dx) ) / X1mu
            dz1 = torch.where(has_lb, dz1, torch.zeros_like(dz1))            # [B,n]
            #dz1 = dz1 * alpha

            dz2 = -( z2 * (x2 - dx + muB_bn) - (muB_bn * z2E) + muB_bn * (- x2E + x2 - dx) ) / X2mu
            dz2 = torch.where(has_ub, dz2, torch.zeros_like(dz2))            # [B,n]
            #dz2 = dz2 * alpha

            # Δy, Δs, Δw1 (only if inequalities exist)
            if m_ineq > 0:
                # J_f Δx
                Jdx = torch.bmm(Gb, dx.unsqueeze(-1)).squeeze(-1)            # [B,m]
                # Δy = (D_Y + D_W)^{-1} ( -grad_y - J_f Δx )
                dy = (-grad_y - Jdx) / (DY + DW_combo + eps)                  # [B,m]
                #dy = - p_y * grad_y 
                # Δs = - D_W ( y + Δy − π^W )
                #ds = - DW_combo * (y + dy - piW)                              # [B,m] 
                ds = - p_s * grad_s
                # Δw1 = -(S1^μ)^{-1} [ w1 ((s + Δs) − s_lb + μ^B) − μ^B w1^E + μ^B (s1^E − s + Δs) ]
                term_w1 = (w1 * ((s + ds) - s_lb + muB_m)) - (muB_m * w1E) + (muB_m * (- s1E + s + ds))
                dw1 = -(term_w1 / S1mu) * mask_s_lb                           # [B,m]

                # dy = dy * alpha
                # ds = ds * alpha
                # dw1 = dw1 * alpha
            else:
                dy = ds = dw1 = None


            # Δv (equality multiplier) via \hat{π}^V
            # if meq and meq > 0:
            #     Ax_next = torch.bmm(Ab, (x + dx).unsqueeze(-1)).squeeze(-1)  # [B,meq]
            #     dvv = vE - (Ax_next - bB) / (DA + eps) - v                  # [B,meq]   (DA = μ^A I)
            #     # dvv = alpha * dvv
            #     vv_new = v + dvv
            # else:
            #     vv_new = v     
            
            if meq and meq > 0:
                Adx = torch.bmm(Ab, dx.unsqueeze(-1)).squeeze(-1)            # [B,m]
                # Δy = (D_Y + D_W)^{-1} ( -grad_y - J_f Δx )
                dvv = (-grad_vv - Adx) / (DV + eps)                  # [B,m]    
                vv_new = v + dvv                                           # pass-through if no equalities
            else:
                vv_new = None

            # ===================== 3) UPDATES =====================
            x_new  = x  + dx
            z1_new = z1 + dz1
            z2_new = z2 + dz2

            if m_ineq > 0:
                y_new  = y  + dy
                #y_new = - p_y * grad_y 
                s_new  = s  + ds
                #s_new = - p_s * grad_s
                w1_new = w1 + dw1
            else:
                y_new, s_new, w1_new = y, s, w1
            
            # print("ds: ", ds)
            
            # print("m_ineq: ", m_ineq)
            # print(m_ineq > 0)
            
            # print("s: ", s)
            # print("s_new: ", s_new)


            # # ===================== 4) CONCAT INTO v_new =====================
            # # Include every block in the state vector (order: x | z1 | z2 | y | s | w1 | v)
            # v_new = torch.cat([
            #     x_new.unsqueeze(-1),       # [B,n,1]
            #     z1_new.unsqueeze(-1),      # [B,n,1]
            #     z2_new.unsqueeze(-1),      # [B,n,1]
            #     (y_new.unsqueeze(-1)  if m_ineq > 0 else torch.empty(B, 0, 1, device=x.device)),
            #     (s_new.unsqueeze(-1)  if m_ineq > 0 else torch.empty(B, 0, 1, device=x.device)),
            #     (w1_new.unsqueeze(-1) if m_ineq > 0 else torch.empty(B, 0, 1, device=x.device)),
            #     (vv_new.unsqueeze(-1) if (meq and meq > 0) else torch.empty(B, 0, 1, device=x.device))
            # ], dim=1)

            if self.device == 'cpu':
                print("grad_x: ", grad_x.abs().mean().item())
                print("grad_z1: ", grad_z1.abs().mean().item())
                print("grad_z2: ", grad_z2.abs().mean().item())
                if problem.num_ineq > 0:
                    print("grad_y: ", grad_y.abs().mean().item())
                    print("grad_s: ", grad_s.abs().mean().item())
                    print("grad_w1: ", grad_w1.abs().mean().item())
                if problem.num_eq > 0:
                    print("grad_v: ", grad_vv.abs().mean().item())


            w2_new = None
            # project all 4 blocks
            if problem.num_eq > 0:
                vv_new = vv_new.unsqueeze(-1)
                vE = vE.unsqueeze(-1)
            if problem.num_ineq > 0:
                s = s.unsqueeze(-1)
                s_new = s_new.unsqueeze(-1)
                s1E = s1E.unsqueeze(-1)
                y = y.unsqueeze(-1)
                y_new = y_new.unsqueeze(-1)
                yE = yE.unsqueeze(-1)
                w1 = w1.unsqueeze(-1)
                w1_new = w1_new.unsqueeze(-1)
                w1E = w1E.unsqueeze(-1)
            
            x, s, y, v, z1, z2, w1 = project_step(problem, 
                x.unsqueeze(-1), s,       # add s1 and s2 if there are bounds for inequality
                y, v, z1.unsqueeze(-1), z2.unsqueeze(-1), w1, w2,   # add w1 and w2 instead of w if there are bounds for inequality
                x_new.unsqueeze(-1), s_new,       # add s1 and s2 if there are bounds for inequality
                y_new, vv_new, z1_new.unsqueeze(-1), z2_new.unsqueeze(-1), w1_new, w2_new, 
                x1E.unsqueeze(-1), x2E.unsqueeze(-1), s1E, s2E, yE, vE, z1E.unsqueeze(-1), z2E.unsqueeze(-1), w1E, w2E,  # add w1E and w2E instead of wE if there are bounds for inequality
                muP,
                muB, muA
            )

            if problem.num_eq > 0:
                vv_new = vv_new.squeeze(-1)
                vE = vE.squeeze(-1)
            
            if problem.num_ineq > 0:
                s = s.squeeze(-1)
                s_new = s_new.squeeze(-1)
                s1E = s1E.squeeze(-1)
                y = y.squeeze(-1)
                y_new = y_new.squeeze(-1)
                yE = yE.squeeze(-1)
                w1 = w1.squeeze(-1)
                w1_new = w1_new.squeeze(-1)
                w1E = w1E.squeeze(-1)

            # print(x.shape)
            # print(lb.shape)
            # raw_x1 = (x - lb.unsqueeze(-1))            # [B,n,1]
            # x1     = torch.where(mask_lb, raw_x1,
            #                     torch.zeros_like(raw_x1))

            # raw_x2 = (ub.unsqueeze(-1) - x)         # [B,n,1]
            # x2     = torch.where(mask_ub, raw_x2,
            #                     torch.zeros_like(raw_x2))


            # # unpack back to flat
            # x   = x.squeeze(-1)   if x.dim()==3   else x             # [B,n]
            # z1 = z1.squeeze(-1)  if z1.dim()==3  else z1            # [B,n]
            # z2 = z2.squeeze(-1)  if z2.dim()==3  else z2            # [B,n]
            # v = v.squeeze(-1)  if v.dim()==3  else v            # [B,n]
            # s   = s.squeeze(-1)   if s.dim()==3   else s         # [B,m]
            # y   = y.squeeze(-1)   if y.dim()==3   else y         # [B,m]
            # w1  = (w1.squeeze(-1) if (w1 is not None and w1.dim()==3) else w1)  # [B,m] or None 
 

            # grad_x, grad_s, grad_y, grad_vv, grad_z1, grad_z2, grad_w1 = problem.merit_grad_M(
            #     x, s,      
            #     y, v, z1, z2, w1, w2,    
            #     x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            #     muP, muB, muA, bad_x1, bad_x2
            # )

            # if grad_x is not None: 
            #     grad_x = grad_x.view(B,   n)
            # if grad_z1 is not None:
            #     grad_z1 = grad_z1.view(B, n)
            # if grad_z2 is not None:
            #     grad_z2= grad_z2.view(B, n)
            # if grad_s is not None: 
            #     grad_s = grad_s.view(B,   n)
            # if grad_y is not None:
            #     grad_y = grad_y.view(B, n)
            # if grad_vv is not None:
            #     grad_vv= grad_vv.view(B, n)
            # if grad_w1 is not None:
            #     grad_w1= grad_w1.view(B, n)


            # computing your step‐loss on the projected point 
            M_proj_new = problem.merit_M(
                x, s,       # add s1 and s2 if there are bounds for inequality
                y, v, z1, z2, w1, w2,     # add w1 and w2 instead of w if there are bounds for inequality
                x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E,  # add w1E and w2E instead of wE if there are bounds for inequality
                muP,
                muB, muA, bad_x1, bad_x2
            )

            if self.device == 'cpu':
                f_val, lb_1, lb_2, lb_3, lb_4, AL_1, ub_1, ub_2, ub_3, ub_4, AL_2, e1, e2, e3 = problem.merit_M_indi(x, s,       # add s1 and s2 if there are bounds for inequality
                    y, v, z1, z2, w1, w2,     # add w1 and w2 instead of w if there are bounds for inequality
                    x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E,  # add w1E and w2E instead of wE if there are bounds for inequality
                    muP,
                    muB, muA, bad_x1, bad_x2)
                print(M_proj_new[M_proj_new > 1e8])
                print(len(M_proj_new[M_proj_new > 1e8]))
                indices = torch.nonzero(M_proj_new > 1e8, as_tuple=False).squeeze()
                print(indices)
                for i in range(min(5, lb.shape[0])):
                    print(f"Problem {i}:")
                    print(f"  muB {i} =  {muB[i]}")
                    print(f"  muP {i} =  {muP[i]}")
                    print(f"  x {i} =  {x[i]}")
                    print(f"  merit_M {i} =  {M_proj_new[i]}")
                    print(f"  f_val {i} =  {f_val[i]}")
                    print(f"  lb_1 {i} =  {lb_1[i]}")
                    print(f"  lb_2 {i} =  {lb_2[i]}")
                    print(f"  lb_3 {i} =  {lb_3[i]}")
                    print(f"  lb_4 {i} =  {lb_4[i]}")
                    print(f"  AL_1 {i} =  {AL_1[i]}")
                    print(f"  ub_1 {i} =  {ub_1[i]}")
                    print(f"  ub_2 {i} =  {ub_2[i]}")
                    print(f"  ub_3 {i} =  {ub_3[i]}")
                    print(f"  ub_4 {i} =  {ub_4[i]}")
                    print(f"  AL_2 {i} =  {AL_2[i]}")
                    print(f"  e1 {i} =  {e1[i]}")
                    print(f"  e2 {i} =  {e2[i]}")
                    print(f"  e3 {i} =  {e3[i]}")
                # indices where merit exceeds threshold
                thr = 1e8
                bad_idx = torch.nonzero(M_proj_new > thr, as_tuple=False).flatten()

                bad_idx = bad_idx[bad_idx != 0]

                print(f"indices over {thr:g}:", bad_idx.tolist())
                print("count:", bad_idx.numel())

                # print at most 5 of those "bad" problems
                for i in bad_idx[:5].tolist():
                    print(f"Problem {i}:")
                    print(f"  muB {i} =  {muB[i]}")
                    print(f"  muP {i} =  {muP[i]}")
                    print(f"  merit_M {i} =  {M_proj_new[i]}")
                    print(f"  f_val {i} =  {f_val[i]}")
                    print(f"  lb_1 {i} =  {lb_1[i]}")
                    print(f"  lb_2 {i} =  {lb_2[i]}")
                    print(f"  lb_3 {i} =  {lb_3[i]}")
                    print(f"  lb_4 {i} =  {lb_4[i]}")
                    print(f"  AL_1 {i} =  {AL_1[i]}")
                    print(f"  ub_1 {i} =  {ub_1[i]}")
                    print(f"  ub_2 {i} =  {ub_2[i]}")
                    print(f"  ub_3 {i} =  {ub_3[i]}")
                    print(f"  ub_4 {i} =  {ub_4[i]}")
                    print(f"  AL_2 {i} =  {AL_2[i]}")
                    print(f"  e1 {i} =  {e1[i]}")
                    print(f"  e2 {i} =  {e2[i]}")
                    print(f"  e3 {i} =  {e3[i]}")

                    # for j in range(lb.shape[1]):
                    #     print(f"  x {j} =  {x[i, j]}")
            
            x   = x.squeeze(-1)   if x.dim()==3   else x             # [B,n]
            z1 = z1.squeeze(-1)  if z1.dim()==3  else z1            # [B,n]
            z2 = z2.squeeze(-1)  if z2.dim()==3  else z2            # [B,n]
            if problem.num_eq > 0:
                v = v.squeeze(-1)  if v.dim()==3  else v            # [B,n]
            if problem.num_ineq > 0:
                s   = s.squeeze(-1)   if s.dim()==3   else s         # [B,m]
                y   = y.squeeze(-1)   if y.dim()==3   else y         # [B,m]
                w1  = (w1.squeeze(-1) if (w1 is not None and w1.dim()==3) else w1)  # [B,m] or None 


            M_i = M_proj_new.squeeze(-1).squeeze(-1)  # [B]
            step_losses.append(M_i)

            if self.device == 'cpu':
                print("M_proj_new: ", M_proj_new.mean())

            total = total + M_proj_new
        
        x   = x.unsqueeze(-1)   if x.dim()==2   else x             # [B,n]
        z1 = z1.unsqueeze(-1)  if z1.dim()==2  else z1            # [B,n]
        z2 = z2.unsqueeze(-1)  if z2.dim()==2  else z2            # [B,n]
        if problem.num_eq > 0:
            v = v.unsqueeze(-1)  if v.dim()==2  else v            # [B,n]
        if problem.num_ineq > 0:
            s   = s.unsqueeze(-1)   if s.dim()==2   else s         # [B,m]
            y   = y.unsqueeze(-1)   if y.dim()==2   else y         # [B,m]
            w1  = (w1.unsqueeze(-1) if (w1 is not None and w1.dim()==2) else w1)  # [B,m] or None 

        
        # x   = x.squeeze(-1)   if x.dim()==3   else x             # [B,n]
        # z1 = z1.squeeze(-1)  if z1.dim()==3  else z1            # [B,n]
        # z2 = z2.squeeze(-1)  if z2.dim()==3  else z2            # [B,n]
        # v = v.squeeze(-1)  if v.dim()==3  else v            # [B,n]
        # s   = s.squeeze(-1)   if s.dim()==3   else s         # [B,m]
        # y   = y.squeeze(-1)   if y.dim()==3   else y         # [B,m]
        # w1  = (w1.squeeze(-1) if (w1 is not None and w1.dim()==3) else w1)  # [B,m] or None 

        all_losses = torch.stack(step_losses, dim=0)  # [K_inner, B]
        step_loss_vector = all_losses.sum(dim=0)      # [B]

        # finally return the blended variables, the step‐loss, and new LSTM/r states
        return (x, s, y, v, z1, z2, w1, w2), step_loss_vector, ((h_new_x, c_new_x), (h_new_s, c_new_s), r_k)


