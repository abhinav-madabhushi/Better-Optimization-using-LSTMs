import torch, torch.nn as nn, torch.nn.functional as F
import time

class PS_L20_LSTM(nn.Module):
    # ------------------------------------------------------------
    def __init__(self, problem, hidden_size=16, num_layers=1, K_inner = 1, device='cuda' if torch.cuda.is_available() else 'cpu', fixedmuNumber = 15): 
        super().__init__()
        # self.input_size = 0
        # self.output_size = 0
        # var = problem.Q.shape[1] # times M here, and also need to think how we are going to input Q, p, lb, and ub. Maybe just expand it M times
        # #self.input_size = var*var + 3*var
        # self.input_size = self.input_size + (2*var)
        # #self.output_size = self.output_size + (var)
        # #self.output_size = self.output_size + ((var)*(var + 1))//2
        # # k = 5
        # # self.output_size = self.output_size + (3*var)+ (3*k*var)
        # self.output_size = self.output_size + (var)
        # flag_muB = False
        # if (problem.num_lb != 0) and (problem.num_ub != 0):
        #     self.input_size = self.input_size + (12*var)
        #     #self.output_size = self.output_size + (2*var)
        #     flag_muB = True                         # ensure Python int
        # m = 3 * var                                  # alias for clarity
        # p_dim = m * (m + 1) // 2      
        # #self.output_size = self.output_size + p_dim
        # #self.output_size = 3*self.output_size
        # self.output_size = self.output_size + (4*3*var)
        # # can increase size of the input using other if conditions if other variables exist
        # if flag_muB:
        #     self.input_size = self.input_size + 1 # for muB

        self.K_inner = K_inner

        self.fixedmuNumber = fixedmuNumber

        self.device = device

        # rebuild your LSTM with the correct input_size
        # self.lstm = nn.LSTM(
        #     input_size  = self.input_size, 
        #     hidden_size = hidden_size,
        #     num_layers  = num_layers,
        #     batch_first = True,
        # )

        # # MLP produces 5N sized output,
        # # sizes = [
        # #     n,        # p_x
        # #     m_tot,    # p_s
        # #     m_tot,    # p_y
        # #     m_tot,    # p_w
        # #     N, N, N, N  # b, a, b1, b2
        # # ]
        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, self.output_size)
        # )

        self.px_feat_dim = 15
        self.lstm_px = nn.LSTM(
            input_size  = self.px_feat_dim,   # independent of n
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
        )
        self.px_head = nn.Sequential(
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
            x, x1, x2, s,       # add s1 and s2 if there are bounds for inequality
            y, v, z1, z2, w,    # add w1 and w2 instead of w if there are bounds for inequality
            x1E, x2E, sE, yE, vE, z1E, z2E, wE, # add w1E and w2E instead of wE if there are bounds for inequality
            muP, muB,
            r_k,                                  # history‐encoder [B, N+2]
            states, bad_x1, bad_x2, bad_z1, bad_z2,  M_max, n_vec,                           # (h_prev, c_prev) for nn.LSTM
            project_step, outer, print_level):
        
        # add for loop here to do multiple iterations of the forward pass before backpropogating 
        #ds_start_iter = time.perf_counter()

        # define the sizes
        B, n     = x.shape
        m_ineq   = problem.num_ineq
        m_eq     = problem.num_eq, 
        N        =  3*n # will need to change if more variables are added

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

            # print(z1.shape)
            # print(x1.shape)
            # z1 = z1.squeeze(-1)
            # z2 = z2.squeeze(-1)

            # x1 = x1.squeeze(-1)
            # x2 = x2.squeeze(-1)

            if x1 is not None:
                slack1 = x1 + muB        # should stay > 0

                # 2) build a mask of infeasible entries
                bad_x1 = (slack1 <= 0)   # shape [B,n,1], True wherever x1+μB ≤ 0

                slackz1 = (z1 + muB)    # [B,n]

                # build masks of the exact indices that went non-positive
                bad_z1 = (slackz1 <= 0)  # [B,n,1]

                # Option A: out-of-place with torch.where
                z1 = torch.where(bad_z1, torch.zeros_like(z1), z1) 
            if x2 is not None:
                slack2 = x2 + muB        # should stay > 0
                bad_x2 = (slack2 <= 0)   # shape [B,n,1], True wherever x2+μB ≤ 0
                slackz2 = (z2 + muB)    # [B,n]
                bad_z2 = (slackz2 <= 0)  # [B,n,1]
                z2 = torch.where(bad_z2, torch.zeros_like(z2), z2)    
            #print(f"Total infeasible variables wrt to x2 in batch: {total_infeas}")

            # slackz1 = (z1.squeeze(-1) + muB)    # [B,n]
            # slackz2 = (z2.squeeze(-1) + muB)    # [B,n]

            # # build masks of the exact indices that went non-positive
            # bad_z1 = (slackz1 <= 0).unsqueeze(-1)  # [B,n,1]
            # bad_z2 = (slackz2 <= 0).unsqueeze(-1)  # [B,n,1]

            # # Option A: out-of-place with torch.where
            # z1 = torch.where(bad_z1, torch.zeros_like(z1), z1)
            # z2 = torch.where(bad_z2, torch.zeros_like(z2), z2)

            # Mtest = problem.Mtest(x, x1, x2, s,      
            #     y, v, z1, z2, w,    
            #     x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
            #     muP, muB, M_max, bad_x1, bad_x2, bad_z1, bad_z2)   

            # ds_start_iter = time.perf_counter() 

            # obtaining each of the gradients
            grad_x, grad_s, grad_y, grad_vv, grad_z1, grad_z2, grad_w = problem.merit_grad_M(
                x, x1, x2, s,       # add s1 and s2 if there are bounds for inequality
                y, v, z1, z2, w,    # add w1 and w2 instead of w if there are bounds for inequality
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, # add w1E and w2E instead of wE if there are bounds for inequality
                muP,
                muB, bad_x1, bad_x2, bad_z1, bad_z2
            )
        
            # flatten gradients
            if grad_x is not None: 
                grad_x = grad_x.view(B,   n)
            if grad_z1 is not None:
                grad_z1 = grad_z1.view(B, n)
            if grad_z2 is not None:
                grad_z2 = grad_z2.view(B, n)

            # ds_end_iter = time.perf_counter()
            # print("Time elapsed: ", ds_end_iter - ds_start_iter)


            # ensure grads can flow
            # x  = x.requires_grad_(True)
            # x1  = x1.requires_grad_(True)
            # x2  = x2.requires_grad_(True)
            # z1 = z1.requires_grad_(True)
            # z2 = z2.requires_grad_(True) 

            M = problem.merit_M(x, x1, x2, s,       # add s1 and s2 if there are bounds for inequality
                y, v, z1, z2, w,    # add w1 and w2 instead of w if there are bounds for inequality
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, # add w1E and w2E instead of wE if there are bounds for inequality
                muP,
                muB, bad_x1, bad_x2, bad_z1, bad_z2)


            # keep the graph if you’ll call backward later in the step
            # gx_raw, gx1_raw, gx2_raw, gz1_raw, gz2_raw = torch.autograd.grad(
            #     outputs=M.sum(),
            #     inputs=[x, x1, x2, z1, z2],
            #     retain_graph=True,
            #     allow_unused=True,   # in case a var is unused under a mask
            #     create_graph=False
            # )

            # replace Nones by zeros for clean printing
            # def nz(g, like): return g if g is not None else torch.zeros_like(like)
            # gx_raw, gx1_raw, gx2_raw, gz1_raw, gz2_raw = nz(gx_raw, x), nz(gx1_raw, x1), nz(gx2_raw, x2), nz(gz1_raw, z1), nz(gz2_raw, z2)

            # print("‖∂M/∂x‖∞ :", gx_raw.abs().amax().item())
            # print("‖∂M/∂x1‖∞ :", gx1_raw.abs().amax().item())
            # print("‖∂M/∂x2‖∞ :", gx2_raw.abs().amax().item())
            # print("‖∂M/∂z1‖∞:", gz1_raw.abs().amax().item())
            # print("‖∂M/∂z2‖∞:", gz2_raw.abs().amax().item())


            # # pad s, y, and w to the correct length
            # if m_eq>0:
            #     pad    = x_flat.new_zeros((B, m_eq))
            #     s_full = torch.cat([s_flat, pad], dim=1)   # [B,m_tot]
            #     y_full = torch.cat([y_flat, pad], dim=1)
            #     w_full = torch.cat([w_flat, pad], dim=1)
            # else:
            #     s_full, y_full, w_full = s_flat, y_flat, w_flat

            m = 3 * n

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
            #     mask_x = torch.ones(B, n, device=self.device)
            #     mask_z1 = mask_x
            #     mask_z2 = mask_x
            #     mask_v = torch.ones(B, m, device=self.device)
            #     mask_H = torch.ones(B, m, m, device=self.device)

            # assume x, z1, z2 are all [B,n] after squeezing
            # and grad_x, grad_z1, grad_z2 are [B,n] too

            # def z_norm(tensor, mask=None, eps=1e-6, across: str = "feature"):
            #     # accept [B,n] or [B,n,1]
            #     squeeze_back = (tensor.dim()==3 and tensor.size(-1)==1)
            #     x = tensor.squeeze(-1) if squeeze_back else tensor              # [B,n]

            #     if across in (None, "none"):
            #         y = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            #         return y.unsqueeze(-1) if squeeze_back else y

            #     if mask is not None:
            #         m = mask.float()                                            # [B,n]
            #         m_sum = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
            #         mean = (x * m).sum(dim=-1, keepdim=True) / m_sum
            #         var  = ((x - mean)**2 * m).sum(dim=-1, keepdim=True) / m_sum
            #     else:
            #         mean = x.mean(dim=-1, keepdim=True)
            #         var  = (x - mean).pow(2).mean(dim=-1, keepdim=True)

            #     std = (var.clamp_min(eps)).sqrt()
            #     y = (x - mean) / std
            #     y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
            #     return y.unsqueeze(-1) if squeeze_back else y



            
            # # why did I do this? Normalize so that LSTM will work better 

            # # normalize primal blocks
            # x_n  = z_norm(x,  mask_x)
            # x1_n = z_norm(x1, mask_x) if x1 is not None else None
            # x2_n = z_norm(x2, mask_x) if x2 is not None else None
            # z1_n = z_norm(z1, mask_z1) if z1 is not None else None
            # z2_n = z_norm(z2, mask_z2) if z2 is not None else None

            # grad_x_n  = z_norm(grad_x,  mask_x)  if grad_x  is not None else None
            # grad_z1_n = z_norm(grad_z1, mask_z1) if grad_z1 is not None else None
            # grad_z2_n = z_norm(grad_z2, mask_z2) if grad_z2 is not None else None

            # x1E_n = z_norm(x1E, mask_x)  if x1E is not None else None
            # x2E_n = z_norm(x2E, mask_x)  if x2E is not None else None
            # z1E_n = z_norm(z1E, mask_z1) if z1E is not None else None
            # z2E_n = z_norm(z2E, mask_z2) if z2E is not None else None



            # # build the state vector
            # parts = []
            # # helper to squeeze and register a block
            # def _add_block(var, name):
            #     if var is not None:
            #         flat = var.squeeze(-1)       # [B, dim]
            #         parts.append(flat)

            # # always include x
            # _add_block(x,  'x')
            # _add_block(x1,  'x1')
            # _add_block(x2,  'x2')

            # _add_block(z1, 'z1')   # lower‐dual
            # _add_block(z2, 'z2')   # upper‐dual

            # # inequality slacks/duals
            # _add_block(s,  's')    # ineq‐slack
            # _add_block(w,  'w')    # ineq‐dual
            # _add_block(y,  'y')    # ineq‐multiplier

            # # equality multiplier
            # _add_block(v,  'v')    # eq‐multiplier

            # # creating v0 according to the math
            # v_k = torch.cat(parts, dim=1)
            # #print(len(parts))  
            # #print(parts)  
            # #print(v)

            # parts = []
            # x1E = x1E 
            # x2E = x2E
            # z1E = z1E 
            # z2E = z2E 
            # # helper to squeeze and register a block
            # def _add_block(var, name):
            #     if var is not None:
            #         flat = var.squeeze(-1)       # [B, dim]
            #         parts.append(flat)

            # # always include x
            # _add_block(x1E,  'x1E')
            # _add_block(x2E,  'x2E')

            # _add_block(z1E, 'z1E')   # lower‐dual
            # _add_block(z2E, 'z2E')   # upper‐dual

            # # inequality slacks/duals
            # _add_block(sE,  'sE')    # ineq‐slack
            # _add_block(wE,  'wE')    # ineq‐dual
            # _add_block(yE,  'yE')    # ineq‐multiplier

            # # equality multiplier
            # _add_block(vE,  'vE')    # eq‐multiplier

            # # creating v0 according to the math
            # v_kE = torch.cat(parts, dim=1)

            # parts = []

            # def _add_grad_block(g, name):
            #     if g is not None:
            #         # [B,d,1] → [B,d]
            #         flat = g.squeeze(-1) if g.dim()==3 else g
            #         parts.append(flat)

            # # primal variables
            # _add_grad_block(grad_x,  'grad_x')
            # _add_grad_block(grad_z1, 'grad_z1')
            # _add_grad_block(grad_z2, 'grad_z2')

            # # inequality blocks
            # _add_grad_block(grad_s,  'grad_s')
            # _add_grad_block(grad_w,  'grad_w')
            # _add_grad_block(grad_y,  'grad_y')

            # # equality multipliers
            # _add_grad_block(grad_vv,  'grad_vv')

            # # now cat
            # grad_v = torch.cat(parts, dim=1)   # [B, N]
            #print(len(parts))

            #print(grad_v.shape)
            #print(v_k.shape)

            if (self.device == 'cpu') & (print_level == 3):
                print("x: ", x.mean().item())
                print("x1: ", x1.mean().item())
                print("x2: ", x2.mean().item())
                print("z1: ", z1.mean().item())
                print("z2: ", z2.mean().item())

                # print("x: ", x)
                # print("x1: ", x1)
                # print("x2: ", x2)
                # print("z1: ", z1)
                # print("z2: ", z2)

            # mask_lb = torch.isfinite(problem.lb)   # [B,n,1], True where ℓ_j > -∞
            # mask_ub = torch.isfinite(problem.ub)   # [B,n,1], True where u_j < +∞

            #total_start_new = time.perf_counter()
            #ds_start = time.perf_counter()


            # B, n = problem.Q.shape[:2]                       # Q is [B , n , n]

            # Q_flat = problem.Q.reshape(B, n*n)               # [B , n²]
            # p_flat = problem.p.reshape(B, n)                 # [B , n]
            lb     = problem.lb.reshape(B, n)                # [B , n]
            ub     = problem.ub.reshape(B, n)    

            # Q_flat  = z_norm(Q_flat)  if Q_flat  is not None else None
            # p_flat  = z_norm(p_flat)  if p_flat  is not None else None
            # lb  = z_norm(lb)  if lb  is not None else None
            # ub  = z_norm(ub)  if ub  is not None else None

            # v_k  = v_k  * torch.cat([mask_x, mask_x, mask_x, mask_z1, mask_z2], dim=1)
            # grad_v = grad_v * torch.cat([mask_x, mask_z1, mask_z2], dim=1)


            eps = 1e-12

            if torch.is_tensor(muB):
                muBv = muB.view(B, 1)
            else:
                muBv = torch.full((B, 1), float(muB), device=self.device)

            invD1 = (z1 + muBv) / (x1 + muBv + eps)               # [B, n]
            invD2 = (z2 + muBv) / (x2 + muBv + eps)               # [B, n]

            inv_cap = 1e2  # tune

            invD1 = invD1.clamp_min(0.0).clamp_max(inv_cap)
            invD2 = invD2.clamp_min(0.0).clamp_max(inv_cap)

            #print(invD1.shape)

            # # compiling lstm input
            # lstm_in = torch.cat([
            #     #Q_flat, p_flat, lb, ub,
            #     v_k,
            #     grad_v, 
            #     v_kE, 
            #     invD1, 
            #     invD2, 
            #     #muP.expand(B,1),
            #     muB
            # ], dim=1)

            # # run through LSTM
            # o_seq, (h_new, c_new) = self.lstm(
            #     lstm_in.unsqueeze(1),    # [B,1,N+2]
            #     states                   # (h_prev, c_prev)
            # )
            # o_k = o_seq.squeeze(1)       # [B, hidden_size]

            # states = (h_new, c_new)

            # out = self.mlp(o_k)

            # m = 3*n
            # p_dim = m * (m + 1) // 2

            # k = 5
            # sizes = (
            #     #p_dim, 
            #     #n, n, n, 
            #     #(n*(n+1))//2, 
            #     n, 
            #     N, N, N, N  # b, a, b1, b2
            # )
            # p_x, b, a, b1, b2 = torch.split(out, sizes, dim=1)

            # p_x = torch.sigmoid(p_x)

            # p_z1 = torch.sigmoid(p_z1)
            # p_z2 = torch.sigmoid(p_z2)

            muB_bn = muB.view(B, 1).expand(B, n)

            def _sq(v):
                if v is None:
                    return None
                return v.squeeze(-1)

            x  = _sq(x)
            x1 = _sq(x1)
            x2 = _sq(x2)
            z1 = _sq(z1)
            z2 = _sq(z2)
            grad_x  = _sq(grad_x)
            grad_z1 = _sq(grad_z1)
            grad_z2 = _sq(grad_z2)
            x1E = _sq(x1E)
            x2E = _sq(x2E)
            z1E = _sq(z1E)
            z2E = _sq(z2E)




            # choose the features you want per coordinate (order must match px_feat_dim)
            feat_px = torch.stack([
                x, 
                x1, 
                x2, 
                z1, 
                z2, 
                grad_x,
                grad_z1,
                grad_z2,
                x1E,
                x2E,
                z1E,
                z2E,
                muB_bn, 
                invD1, 
                invD2
            ], dim=-1) 

            # def _usq(v):
            #     if v is None:
            #         return None
            #     return v.unsqueeze(-1)

            # x  = _usq(x)
            # x1 = _usq(x1)
            # x2 = _usq(x2)
            # z1 = _usq(z1)
            # z2 = _usq(z2)
            # grad_x  = _usq(grad_x)
            # grad_z1 = _usq(grad_z1)
            # grad_z2 = _usq(grad_z2)
            # x1E = _usq(x1E)
            # x2E = _usq(x2E)
            # z1E = _usq(z1E)
            # z2E = _usq(z2E)



            BN   = B * n
            inp  = feat_px.reshape(BN, self.px_feat_dim)        # [BN, F]
            inp  = inp.unsqueeze(1)



            if n_vec is not None:
                mask = (torch.arange(n, device=x.device)[None,:] < n_vec[:,None]).float()
                Y = Y * mask.unsqueeze(-1)
                Y, (h_new, c_new) = self.lstm_px(inp)    
            else:
                Y, (h_new, c_new) = self.lstm_px(inp)          # [B, n, H]

            states = (h_new, c_new)


            p_x = self.px_head(Y).reshape(B, n)

            p_x = torch.abs(p_x)
            # low, high = 1e-5, 1e-2
            # p_x = low + (high - low) * torch.sigmoid(p_x)

            # p_x, p_z1, p_z2 = p.chunk(3, dim=-1)

            # p_x = p_x.reshape(B, n)
            # p_z1 = p_z1.reshape(B, n)
            # p_z2 = p_z2.reshape(B, n)

            # p_x = F.softplus(p_x)   
            # p_z1 = F.softplus(p_z1)   
            # p_z2 = F.softplus(p_z2)    
          

            # if padded, mask p_x too
            if n_vec is not None:
                p_x = p_x * mask
            


            d_x  = - p_x * grad_x #- (p_z1 * invD1 * grad_z1) + (p_z2 * invD2 * grad_z2)
            d_z1 = (-invD1) * ( - p_x * grad_x + grad_z1)
            d_z2 = (-invD2) * (  p_x * grad_x + grad_z2)

            # step_norm = (d_x**2 + d_z1**2 + d_z2**2).sum(dim=1, keepdim=True).sqrt()
            # scale = torch.clamp(1.0 / (step_norm + 1e-12), max=1.0)
            # d_x, d_z1, d_z2 = d_x*scale, d_z1*scale, d_z2*scale

            x_new = x + d_x #+ p_z1 * grad_z1 - p_z2 * grad_z2

            z1_new = z1  + d_z1      # [B, n]
            z2_new = z2 + d_z2      # [B, n]


            # v_new = torch.cat([x_new, z1_new, z2_new], dim=1)

            # ds_elapsed = time.perf_counter() - ds_start
            # print("Time elapsed: ", ds_elapsed)

            #print(p_x)



            # B, total = p.shape
            # m = 3 * n               # total dimension (3n)

            # # --- unpack p ---
            # d_raw = p[:, :m]                  # [B,3n]
            # U_raw = p[:, m:].view(B, m, k)    # [B,3n,k]

            # d = F.softplus(d_raw) + eps       # ensure positive diag

            # # --- build gradient ---
            # g = torch.cat([grad_x, grad_z1, grad_z2], dim=1)  # [B,3n]

            # # (D + U U^T) g
            # Dg = d * g
            # u_t_g = torch.einsum('bmk,bm->bk', U_raw, g)       # [B,k]
            # Uutg  = torch.einsum('bmk,bk->bm', U_raw, u_t_g)   # [B,3n]
            # step  = -(Dg + Uutg)                               # descent direction [B,3n]

            # # --- split and update ---
            # dx, dz1, dz2 = torch.split(step, (n, n, n), dim=1)

            # alpha = 0.01
            # x_new  = x  + alpha * dx
            # z1_new = z1 + alpha * dz1
            # z2_new = z2 + alpha * dz2

            # v_new = torch.cat([x_new, z1_new, z2_new], dim=1)  # [B,3n]

            # B, three_n = p.shape
            # n = three_n // 3
            # m = 3*n
            # eps = 1e-8

            # # ---- split into v1,v2,v3 ------------------------------------
            # v1, v2, v3 = torch.split(p, n, dim=1)  # each [B,n]

            # v1 = v1.unsqueeze(-1)  # [B,n,1]
            # v2 = v2.unsqueeze(-1)
            # v3 = v3.unsqueeze(-1)

            # # ---- build 3x3 block Gram matrix ----------------------------
            # # Each block [B,n,n]
            # B11 = v1 @ v1.transpose(1,2)
            # B12 = v1 @ v2.transpose(1,2)
            # B13 = v1 @ v3.transpose(1,2)
            # B21 = v2 @ v1.transpose(1,2)
            # B22 = v2 @ v2.transpose(1,2)
            # B23 = v2 @ v3.transpose(1,2)
            # B31 = v3 @ v1.transpose(1,2)
            # B32 = v3 @ v2.transpose(1,2)
            # B33 = v3 @ v3.transpose(1,2)

            # Hinv = p.new_zeros(B, m, m)
            # Hinv[:, 0*n:1*n, 0*n:1*n] = B11
            # Hinv[:, 0*n:1*n, 1*n:2*n] = B12
            # Hinv[:, 0*n:1*n, 2*n:3*n] = B13
            # Hinv[:, 1*n:2*n, 0*n:1*n] = B21
            # Hinv[:, 1*n:2*n, 1*n:2*n] = B22
            # Hinv[:, 1*n:2*n, 2*n:3*n] = B23
            # Hinv[:, 2*n:3*n, 0*n:1*n] = B31
            # Hinv[:, 2*n:3*n, 1*n:2*n] = B32
            # Hinv[:, 2*n:3*n, 2*n:3*n] = B33

            # # add ridge for SPD
            # Hinv = Hinv + eps * torch.eye(m, device=p.device).unsqueeze(0)

            # # apply mask
            # Hinv = Hinv * mask_H

            # # ---- gradient vector ----------------------------------------
            # g = torch.cat([grad_x, grad_z1, grad_z2], dim=1) * mask_v   # [B,m]

            # # ---- descent direction --------------------------------------
            # d = -(Hinv @ g.unsqueeze(-1)).squeeze(-1) * mask_v

            # dx, dz1, dz2 = torch.split(d, (n,n,n), dim=1)

            # alpha = 0.01
            # # ---- update variables ---------------------------------------
            # x_new  = (x  + alpha * dx)  * mask_x
            # z1_new = (z1 + alpha * dz1) * mask_z1
            # z2_new = (z2 + alpha * dz2) * mask_z2

            # v_new = torch.cat([x_new, z1_new, z2_new], dim=1)

            # make the step‐sizes positive, can use softplus also here
            # p_x, p_z1, p_z2, b, a, b1, b2 = torch.split(out, sizes, dim=1)
            # p_x = torch.sigmoid(p_x)
            # p_z1 = torch.sigmoid(p_z1)
            # p_z2 = torch.sigmoid(p_z2)



            # a = torch.sigmoid(a)
            # b = torch.sigmoid(b)

            # concatenating the direction
            #p_v    = torch.cat([p_x, p_z1, p_z2], dim=1)   # [B, N]

            # # reshape into batches of matrices
            # p_x_mat  = p_x.view(B, n, n)     # [B, n, n]
            # p_z1_mat = p_z1.view(B, n, n)
            # p_z2_mat = p_z2.view(B, n, n)

            # # prepare your gradient vectors
            # g_x  = grad_x .unsqueeze(-1)     # [B, n, 1]
            # g_z1 = grad_z1.unsqueeze(-1)
            # g_z2 = grad_z2.unsqueeze(-1)

            # # batched mat-vec multiplies
            # dir_x  = torch.bmm(p_x_mat,  g_x ) .squeeze(-1)   # [B, n]
            # dir_z1 = torch.bmm(p_z1_mat, g_z1).squeeze(-1)
            # dir_z2 = torch.bmm(p_z2_mat, g_z2).squeeze(-1)

            # # now update each block
            # x_new  = x  - dir_x
            # z1_new = z1 - dir_z1
            # z2_new = z2 - dir_z2

            # # if you want to pack them back into one v vector:
            # v_new = torch.cat([x_new, z1_new, z2_new], dim=1)  # [B, 3n]



            if (self.device == 'cpu') & (print_level == 3):
                # print("p_x: ", p_x.mean().item())
                # print("p_z1: ", p_z1.mean().item())
                # print("p_z2: ", p_z2.mean().item())
                #print("a: ", a.mean().item())
                #print("b: ", b.mean().item())
                #print("b_1: ", b1.mean().item())
                #print("b_2: ", b2.mean().item())


                print("grad_x: ", grad_x.abs().mean().item())
                # print("grad_x1: ", grad_x.mean().item())
                # print("grad_x2: ", grad_x.mean().item())
                print("grad_z1: ", grad_z1.abs().mean().item())
                print("grad_z2: ", grad_z2.abs().mean().item())

            # b = torch.sigmoid(b)
            # b1 = torch.sigmoid(b1)
            # b2 = torch.sigmoid(b2)

            # parts = []
            # # helper to squeeze and register a block
            # def _add_block(var, name):
            #     if var is not None:
            #         flat = var.squeeze(-1)       # [B, dim]
            #         parts.append(flat)

            # # always include x
            # _add_block(x,  'x')

            # _add_block(z1, 'z1')   # lower‐dual
            # _add_block(z2, 'z2')   # upper‐dual

            # # inequality slacks/duals
            # _add_block(s,  's')    # ineq‐slack
            # _add_block(w,  'w')    # ineq‐dual
            # _add_block(y,  'y')    # ineq‐multiplier

            # # equality multiplier
            # _add_block(v,  'v')    # eq‐multiplier

            # v_k = torch.cat(parts, dim=1)
            #v_k = v_k[~mask_M]

            # parts = []

            # def _add_grad_block(g, name):
            #     if g is not None:
            #         # [B,d,1] → [B,d]
            #         flat = g.squeeze(-1) if g.dim()==3 else g
            #         parts.append(flat)

            # # primal variables
            # _add_grad_block(grad_x,  'grad_x')
            # _add_grad_block(grad_z1, 'grad_z1')
            # _add_grad_block(grad_z2, 'grad_z2')

            # # inequality blocks
            # _add_grad_block(grad_s,  'grad_s')
            # _add_grad_block(grad_w,  'grad_w')
            # _add_grad_block(grad_y,  'grad_y')

            # # equality multipliers
            # _add_grad_block(grad_vv,  'grad_vv')

            # # now cat
            # grad_v = torch.cat(parts, dim=1)   # [B, N]

            # print(v_k.shape)
            # print(p_v.shape)
            # print(grad_v.shape)

            # x_new = x - p_x * grad_x

            # grad_x, grad_s, grad_y, grad_vv, grad_z1, grad_z2, grad_w = problem.merit_grad_M(
            #     x_new, x1, x2, s,       # add s1 and s2 if there are bounds for inequality
            #     y, v, z1, z2, w,    # add w1 and w2 instead of w if there are bounds for inequality
            #     x1E, x2E, sE, yE, vE, z1E, z2E, wE, # add w1E and w2E instead of wE if there are bounds for inequality
            #     muP,
            #     muB, bad_x1, bad_x2, bad_z1, bad_z2
            # )

            # grad_z1 = grad_z1.squeeze(-1)
            # grad_z2 = grad_z2.squeeze(-1)

            # z1_new = z1 - p_z1 * grad_z1
            # z2_new = z2 - p_z2 * grad_z2

            # v_hat = torch.cat([x_new.squeeze(-1), z1_new.squeeze(-1), z2_new.squeeze(-1)], dim=1)

            # Hinv = problem.merit_hess_inv_M(
            #     x, x1, x2, s,       # add s1 and s2 if there are bounds for inequality
            #     y, v, z1, z2, w,    # add w1 and w2 instead of w if there are bounds for inequality
            #     x1E, x2E, sE, yE, vE, z1E, z2E, wE, # add w1E and w2E instead of wE if there are bounds for inequality
            #     muP,
            #     muB
            # )

            # g = grad_v.unsqueeze(-1)                      # [B,3,1]
            # delta = torch.bmm(Hinv, g).squeeze(-1)        # [B,3]

            # # optional global damping  γ ∈ (0,1]  if you want a line-search-like factor
            # gamma = 0.01                                   # or e.g. 0.5
            # v_new = v_k - gamma * delta                   # [B,3]  ← Newton step

            # def finite(name, t):
            #     ok = torch.isfinite(t)
            #     if not ok.all():
            #         bad = ~ok
            #         print(f"[BAD] {name}: NaN={torch.isnan(t).any()}, Inf={torch.isinf(t).any()}")
            #         idx = torch.nonzero(bad, as_tuple=False)[:10]
            #         print(" sample bad vals:", t[bad][:10])
            #         raise RuntimeError(f"{name} non-finite")

            # finite("p_v",    p_v)
            # finite("grad_v", grad_v)
            # finite("v_k",    v_k)
            # step = p_v * grad_v
            # finite("p_v*grad_v", step)

            # attempt to make HESSIAN INVERSE, but does not converge because matrix not SPD

            # B = p.shape[0]
            # # reshape p into 3x3 blocks, each block an n-vector
            # p_blocks = p.view(B, 3, 3, n)         # [B, 3, 3, n]
            # # rows:    [x-row, z1-row, z2-row]
            # # cols:    [g_x,   g_z1,   g_z2]  contributions

            # # stack gradients into a 3×1 block "vector" (each entry is an n-vector)
            # grads = torch.stack([grad_x, grad_z1, grad_z2], dim=1)   # [B, 3, n]

            # # Block-matrix × block-vector with blockwise dot-products:
            # # S[b,i,j] = < p_blocks[b,i,j,:] , grads[b,j,:] >  (scalar)
            # # Result S is a 3x3 scalar matrix per batch
            # S = torch.einsum('bijn,bjn->bij', p_blocks, grads)       # [B, 3, 3]

            # # Multiply by the 3×1 block "vector" (i.e., sum across j)
            # # This yields 3 scalars per batch: one for x, one for z1, one for z2
            # s = S.sum(dim=2)                                         # [B, 3]

            # # Use those scalars to scale each gradient block (per-batch scalar times n-vector)
            # dir_blocks = s.unsqueeze(-1) * grads                     # [B, 3, n]
            # dir_x, dir_z1, dir_z2 = dir_blocks[:,0,:], dir_blocks[:,1,:], dir_blocks[:,2,:]  # each [B, n]

            # # Update (preconditioned, block-coupled scalar step per variable)
            # x_new  = x  - dir_x
            # z1_new = z1 - dir_z1
            # z2_new = z2 - dir_z2

            # # If you keep a concatenated v:
            # v_new = torch.cat([x_new, z1_new, z2_new], dim=1)        # [B, 3n]

            # obtaining each of the gradients

            # Solve for a Newton-like step: H * step = grad_v
            # H = problem.merit_hess_inv_M(  # (your function name says _inv_; here you use it as H)
            #     x, x1, x2, s, y, v, z1, z2, w,
            #     x1E, x2E, sE, yE, vE, z1E, z2E, wE,
            #     muP, muB, bad_x1, bad_x2, bad_z1, bad_z2
            # )
            # H = H.to(dtype=grad_v.dtype, device=grad_v.device)

            # step = torch.linalg.solve(H, grad_v.unsqueeze(-1)).squeeze(-1)  # [B,3n]
            # d = -step                                                        # descent direction

            # # current merit M0 at (x,z1,z2)
            # # (If you already have M from previous lines, set M0 = M and skip recompute.)
            # M0 = problem.merit_M(
            #     x, x1, x2, s, y, v, z1, z2, w,
            #     x1E, x2E, sE, yE, vE, z1E, z2E, wE,
            #     muP, muB, bad_x1, bad_x2, bad_z1, bad_z2
            # )
            # # make sure shapes broadcast (B,1)
            # M0 = M0.view(-1, 1)

            # # Armijo constants
            # c = 1e-4
            # beta = 0.5
            # alpha = 1.0

            # # directional derivative g^T d  (B,1)
            # gTd = (grad_v * d).sum(dim=1, keepdim=True)

            # # require descent (numerically may be ~0; if not descent, fall back to gradient step or skip)
            # # assert (gTd < 0).all(), "Direction must be descent"

            # max_bt = 100
            # for _ in range(max_bt):
            #     v_try  = v_k + alpha * d
            #     x_try  = v_try[:, :n]
            #     z1_try = v_try[:, n:2*n]
            #     z2_try = v_try[:, 2*n:3*n]

            #     # recompute slacks from x_try (fixes your earlier bug that used x)
            #     x1_try = (x_try - lb)
            #     x2_try = (ub - x_try)
            #     # x1_try = torch.where(mask_lb, raw_x1, torch.zeros_like(raw_x1))
            #     # x2_try = torch.where(mask_ub, raw_x2, torch.zeros_like(raw_x2))

            #     M_try = problem.merit_M(
            #         x_try, x1_try, x2_try, s, y, v, z1_try, z2_try, w,
            #         x1E, x2E, sE, yE, vE, z1E, z2E, wE,
            #         muP, muB, bad_x1, bad_x2, bad_z1, bad_z2
            #     ).view(-1, 1)

            #     # Armijo test: M(v_k + α d) ≤ M(v_k) + c α g^T d  (apply to all batch items)
            #     if (M_try <= M0 + c * alpha * gTd).all():
            #         break
            #     alpha *= beta

            # # accept
            # v_new  = v_k + alpha * d
            # x_new  = v_new[:, :n]
            # z1_new = v_new[:, n:2*n]
            # z2_new = v_new[:, 2*n:3*n]





            # raw update for v
            # if outer < 20:
            #     v_new = v_k - 0.5 * grad_v
            # else:
            #v_new = v_k - grad_v * p_v    # [B, N]

            #v_new = v_k - 0.0001 * grad_v

            # Build SPD preconditioner via true Cholesky from packed lower-tri entries in `p`

            # B = p.shape[0]
            # m = 3 * n  # total dimension for [x, z1, z2]

            # # ---- build L from p (lower-triangular), then mask ------------------
            # L = torch.zeros(B, m, m, device=self.device)
            # tri_i, tri_j = torch.tril_indices(m, m, device=self.device)
            # L[:, tri_i, tri_j] = p

            # L = L * mask_H  # zero out padded parts if any

            # # enforce positive diagonal only on valid diagonal entries
            # diag_idx = torch.arange(m, device=self.device)
            # diag_vals = F.softplus(L[:, diag_idx, diag_idx]) + 1e-8                           # [B,m]
            # L[:, diag_idx, diag_idx] = diag_vals * mask_v                                     # zero padded diags

            # # ---- preconditioner H = L L^T, then mask ---------------------------
            # P = L @ L.transpose(1, 2)                 # [B,m,m]
            # P = P * mask_H

            # # ---- gradient vector and masked step -------------------------------
            # g = torch.cat([grad_x, grad_z1, grad_z2], dim=1) * mask_v                          # [B,m]
            # Pg = (P @ g.unsqueeze(-1)).squeeze(-1)                                             # [B,m]
            # d  = -Pg * mask_v                                                                  # [B,m]

            # # ---- split and update -----------------------------------------------
            # alpha = 0.01
            # dx, dz1, dz2 = torch.split(d, (n, n, n), dim=1)

            # x_new  = (x  + alpha * dx)  * mask_x
            # z1_new = (z1 + alpha * dz1) * mask_z1
            # z2_new = (z2 + alpha * dz2) * mask_z2

            # v_new = torch.cat([x_new, z1_new, z2_new], dim=1)  # [B, m]

            # Ensure positivity / numerical safety where appropriate
            # --- assume you have: p [B, n*(n+1)//2], x,x1,x2,z1,z2, grad_x,grad_z1,grad_z2
            # --- masks: mask_x, mask_z1, mask_z2 (each [B,n]); mask_H [B,3n,3n]
            # --- muB: scalar or [B] or [B,1]
            # eps = 1e-12
            # B, n = x1.shape
            # m = 3 * n
            # device = x1.device

            # # ---------- 1) Build S^{-1} = L L^T from p (lower-tri, row-major) ----------
            # L = torch.zeros(B, n, n, device=device)
            # ii, jj = torch.tril_indices(n, n, device=device)      # row-major lower-tri indices
            # L[:, ii, jj] = p                                      # fill lower-tri entries
            # # enforce positive diagonal → SPD
            # diag = torch.arange(n, device=device)
            # L[:, diag, diag] = torch.nn.functional.softplus(L[:, diag, diag]) + eps

            # Sinv = L @ L.transpose(1, 2)                          # [B, n, n]

            # # ---------- 2) Build (D1^z)^{-1}, (D2^z)^{-1} diagonals as vectors ----------
            # # invD1 = (z1 + muB) / (x1 + muB), invD2 = (z2 + muB) / (x2 + muB)
            # if torch.is_tensor(muB):
            #     muBv = muB.view(B, 1)
            # else:
            #     muBv = torch.full((B, 1), float(muB), device=device)

            # invD1 = (z1 + muBv) / (x1 + muBv + eps)               # [B, n]
            # invD2 = (z2 + muBv) / (x2 + muBv + eps)               # [B, n]

            # # helpers for left/right multiplying by diag(vec) without materializing diag
            # def right_mul_diag(M, v):   # M @ diag(v): scale columns
            #     return M * v.unsqueeze(1)                           # [B,n,n] * [B,1,n]
            # def left_mul_diag(v, M):    # diag(v) @ M: scale rows
            #     return M * v.unsqueeze(-1)                          # [B,n,n] * [B,n,1]

            # # ---------- 3) Precompute mixed products for H^{-1} blocks ----------
            # Sinv_invD1       = right_mul_diag(Sinv, invD1)          # S^{-1}(D1)^{-1}
            # Sinv_invD2       = right_mul_diag(Sinv, invD2)          # S^{-1}(D2)^{-1}
            # invD1_Sinv       = left_mul_diag(invD1, Sinv)           # (D1)^{-1}S^{-1}
            # invD2_Sinv       = left_mul_diag(invD2, Sinv)           # (D2)^{-1}S^{-1}
            # invD1_Sinv_invD1 = right_mul_diag(invD1_Sinv, invD1)    # (D1)^{-1}S^{-1}(D1)^{-1}
            # invD2_Sinv_invD2 = right_mul_diag(invD2_Sinv, invD2)    # (D2)^{-1}S^{-1}(D2)^{-1}
            # invD1_Sinv_invD2 = right_mul_diag(invD1_Sinv, invD2)    # (D1)^{-1}S^{-1}(D2)^{-1}
            # invD2_Sinv_invD1 = right_mul_diag(invD2_Sinv, invD1)    # (D2)^{-1}S^{-1}(D1)^{-1}

            # # ---------- 4) Assemble H^{-1} (3x3 block matrix of n×n blocks) ----------
            # Hinv = Sinv.new_zeros(B, m, m)

            # # row 0
            # Hinv[:, 0*n:1*n, 0*n:1*n] =  Sinv
            # Hinv[:, 0*n:1*n, 1*n:2*n] = -Sinv_invD1
            # Hinv[:, 0*n:1*n, 2*n:3*n] =  Sinv_invD2

            # # row 1
            # # middle (1,1) block: (D1)^{-1} + (D1)^{-1} S^{-1} (D1)^{-1}
            # mid11 = invD1_Sinv_invD1.clone()
            # idx = torch.arange(n, device=device)
            # mid11[:, idx, idx] += invD1                               # add diag(invD1) efficiently
            # Hinv[:, 1*n:2*n, 0*n:1*n] = -invD1_Sinv
            # Hinv[:, 1*n:2*n, 1*n:2*n] =  mid11
            # Hinv[:, 1*n:2*n, 2*n:3*n] = -invD1_Sinv_invD2

            # # row 2
            # # middle (2,2) block: (D2)^{-1} + (D2)^{-1} S^{-1} (D2)^{-1}
            # mid22 = invD2_Sinv_invD2.clone()
            # mid22[:, idx, idx] += invD2
            # Hinv[:, 2*n:3*n, 0*n:1*n] =  invD2_Sinv
            # Hinv[:, 2*n:3*n, 1*n:2*n] = -invD2_Sinv_invD1
            # Hinv[:, 2*n:3*n, 2*n:3*n] =  mid22

            # # apply mask if you have padding
            # Hinv = Hinv * mask_H

            # # ---------- 5) Apply step: d = - H^{-1} g, split, mask, update ----------
            # mask_v = torch.cat([mask_x, mask_z1, mask_z2], dim=1)       # [B, 3n]
            # g = torch.cat([grad_x, grad_z1, grad_z2], dim=1) * mask_v   # [B, 3n]

            # step = -(Hinv @ g.unsqueeze(-1)).squeeze(-1) * mask_v       # [B, 3n]
            # dx, dz1_step, dz2_step = torch.split(step, (n, n, n), dim=1)

            # alpha = 0.01
            # x_new  = (x  + alpha * dx)      * mask_x
            # z1_new = (z1 + alpha * dz1_step)* mask_z1
            # z2_new = (z2 + alpha * dz2_step)* mask_z2
            # v_new  = torch.cat([x_new, z1_new, z2_new], dim=1)          # [B, 3n]


            # d00, d01, d02, d10, d11, d12, d20, d21, d22 = torch.split(p, n, dim=1)

            # # Optionally enforce positivity for the "self" blocks (d00,d11,d22) if you want SPD
            # d00 = F.softplus(d00) 
            # d11 = F.softplus(d11) 
            # d22 = F.softplus(d22) 
            # # Cross terms (d01, d02, d10, d12, d20, d21) can remain unconstrained

            # # Gradient concatenated: g = [gx,gz1,gz2], each [B,n]
            # gx, gz1, gz2 = grad_x, grad_z1, grad_z2

            # # Apply block-matrix * vector, but all blocks are diagonal → elementwise products
            # dx  = d00*gx  + d01*gz1 + d02*gz2
            # dz1 = d10*gx  + d11*gz1 + d12*gz2
            # dz2 = d20*gx  + d21*gz1 + d22*gz2

            # # Descent direction
            # dx, dz1, dz2 = -dx, -dz1, -dz2

            # alpha = 0.01
            # # Mask + update
            # x_new  = (gx  + alpha * dx)  * mask_x
            # z1_new = (gz1 + alpha * dz1) * mask_z1
            # z2_new = (gz2 + alpha * dz2) * mask_z2

            # v_new = torch.cat([x_new, z1_new, z2_new], dim=1)



            # x_r = r_k[:, :n] 
            # z1_r = r_k[:, n:(2*n)] 
            # z2_r = r_k[:, 2*n:3*n] 

            # initialize x1, x2, z1, and z2
            lb = problem.lb.view(B, n, 1).expand(B, n, 1)   # [B,n,1]
            ub = problem.ub.view(B, n, 1).expand(B, n, 1)   # [B,n,1]

            # masks for where bounds are finite
            mask_lb = problem.has_lb.unsqueeze(-1)   # [B,n,1], True where ℓ_j > -∞
            mask_ub = problem.has_ub.unsqueeze(-1)    # [B,n,1], True where u_j < +∞

            # --- lower‐bound slack & dual ---
            # slack  x1_j = clamp(x_j - ℓ_j, min=eps)
            # raw_x1 = (x_r.unsqueeze(-1) - lb)             # [B,n,1]
            # x1_r     = torch.where(mask_lb, raw_x1,
            #                     torch.zeros_like(raw_x1))
            # x1_r = x1_r.squeeze(-1)

            # raw_x2 = (ub - x_r.unsqueeze(-1))            # [B,n,1]
            # x2_r     = torch.where(mask_ub, raw_x2,
            #                     torch.zeros_like(raw_x2))
            # x2_r = x2_r.squeeze(-1)

            #print(x2E.shape)


            # obtaining each of the gradients for r_k
            # grad_x_r, grad_s_r, grad_y_r, grad_vv_r, grad_z1_r, grad_z2_r, grad_w_r = problem.merit_grad_M(
            #     x_r.unsqueeze(-1), x1_r.unsqueeze(-1), x2_r.unsqueeze(-1), s,       # add s1 and s2 if there are bounds for inequality
            #     y, v, z1_r.unsqueeze(-1), z2_r.unsqueeze(-1), w,    # add w1 and w2 instead of w if there are bounds for inequality
            #     x1E.unsqueeze(-1), x2E.unsqueeze(-1), sE, yE, vE, z1E.unsqueeze(-1), z2E.unsqueeze(-1), wE, # add w1E and w2E instead of wE if there are bounds for inequality
            #     muP,
            #     muB, bad_x1, bad_x2, bad_z1, bad_z2
            # )
            # # flatten gradients
            # # flatten gradients
            # if grad_x_r is not None: 
            #     grad_x_r = grad_x_r.view(B,   n)
            # if grad_z1_r is not None:
            #     grad_z1_r = grad_z1_r.view(B, n)
            # if grad_z2_r is not None:
            #     grad_z2_r = grad_z2_r.view(B, n)

            # parts = []

            # # primal variables
            # _add_grad_block(grad_x_r,  'grad_x_r')
            # _add_grad_block(grad_z1_r, 'grad_z1_r')
            # _add_grad_block(grad_z2_r, 'grad_z2_r')

            # # inequality blocks
            # _add_grad_block(grad_s_r,  'grad_s_r')
            # _add_grad_block(grad_w_r,  'grad_w_r')
            # _add_grad_block(grad_y_r,  'grad_y_r')

            # # equality multipliers
            # _add_grad_block(grad_vv_r,  'grad_vv_r')

            # # now cat
            # grad_r = torch.cat(parts, dim=1)   # [B, N]


            # updating r_k
            #r_hat  = r_k - p_v * grad_r   # [B, N]

            

            # finding v_k+1 and r_k+1
            #v_new  = (1.0 - b) * v_hat  + (b * r_hat) - b1               
         

            # split v_new 
            # x_new = v_new[:,:n]                      
            # z1_new = v_new[:, n :   2*n]               
            # z2_new = v_new[:, 2*n :  3*n]               
            sk_new = None
            yk_new = None
            vk_new = None
            wk_new = None

            # --- lower‐bound slack & dual ---
            # slack  x1_j = clamp(x_j - ℓ_j, min=eps)


            # make a function here that checks if variable is new, and unpacks accordingly. This is not the best way to do it, it should work with any problem. 

            # print("x_new: ", x_new.mean().item())
            # for i in range(min(5, lb.shape[0])):
            #     print(f"Problem {i}:")
            #     print(f"  muB {i} =  {muB[i]}")
            #     print(f"  Mtest {i} =  {Mtest[i]}")
            #     for j in range(lb.shape[1]):
            #         print(f"  x_new {j} =  {x_new[i, j]}")
            #         print(f"  p_x {j} =  {p_x[i, j]}")
            #         print(f"  grad_x {j} =  {grad_x[i, j]}")
            #         print(f"  grad_z1 {j} =  {grad_z1[i, j]}")
            #         print(f"  grad_z2 {j} =  {grad_z2[i, j]}")
            #     print()



            # project all 4 blocks
            x, s, y, v, z1, z2, w = project_step(problem, 
                x.unsqueeze(-1), s,       # add s1 and s2 if there are bounds for inequality
                y, v, z1.unsqueeze(-1), z2.unsqueeze(-1), w,    # add w1 and w2 instead of w if there are bounds for inequality
                x_new, sk_new,       # add s1 and s2 if there are bounds for inequality
                yk_new, vk_new, z1_new, z2_new, wk_new,
                x1E.unsqueeze(-1), x2E.unsqueeze(-1), sE, yE, vE, z1E.unsqueeze(-1), z2E.unsqueeze(-1), wE, # add w1E and w2E instead of wE if there are bounds for inequality
                muP,
                muB
            )

            raw_x1 = (x - lb)            # [B,n,1]
            x1     = torch.where(mask_lb, raw_x1,
                                torch.zeros_like(raw_x1))

            raw_x2 = (ub - x)         # [B,n,1]
            x2     = torch.where(mask_ub, raw_x2,
                                torch.zeros_like(raw_x2))

            
            
            # x_flat  = x.squeeze(-1)
            # z1_flat = z1.squeeze(-1)
            # z2_flat = z2.squeeze(-1)

            # # 2) concatenate along the variable‐axis → [B,3*n]
            # v_new = torch.cat([x_flat, z1_flat, z2_flat], dim=1)

            #r_k = v_new + a * (v_new - v_k) + b2

            # x_r = r_k[:, :n] 
            # z1_r = r_k[:, n:(2*n)] 
            # z2_r = r_k[:, 2*n:3*n] 

            # x_r, s, y, v, z1_r, z2_r, w = project_step(problem, 
            #     x_r.unsqueeze(-1), s,       # add s1 and s2 if there are bounds for inequality
            #     y, v, z1_r.unsqueeze(-1), z2_r.unsqueeze(-1), w,    # add w1 and w2 instead of w if there are bounds for inequality
            #     x_new, sk_new,       # add s1 and s2 if there are bounds for inequality
            #     yk_new, vk_new, z1_new, z2_new, wk_new,
            #     x1E.unsqueeze(-1), x2E.unsqueeze(-1), sE, yE, vE, z1E.unsqueeze(-1), z2E.unsqueeze(-1), wE, # add w1E and w2E instead of wE if there are bounds for inequality
            #     muP,
            #     muB
            # ) # do i need to project here???

            # r_k = torch.cat([x_r.squeeze(-1), z1_r.squeeze(-1), z2_r.squeeze(-1)], dim=1)

            # unpack back to flat
            x = x.squeeze(-1)   
            x1 = x1.squeeze(-1)  
            x2 = x2.squeeze(-1)   
            z1 = z1.squeeze(-1)   
            z2 = z2.squeeze(-1)  

            # slack1 = x1.squeeze(-1) + muB        # should stay > 0
            # slack2 = x2.squeeze(-1) + muB        # should stay > 0

            # # 2) build a mask of infeasible entries
            # bad_x1 = (slack1 <= 0)   # shape [B,n,1], True wherever x1+μB ≤ 0
            # bad_x2 = (slack2 <= 0)   # shape [B,n,1], True wherever x2+μB ≤ 0

            # # total_infeas = int(bad_x1.sum().item())
            # # #print(f"Total infeasible variables wrt to x1 in batch: {total_infeas}")
            # # total_infeas = int(bad_x2.sum().item())
            # # #print(f"Total infeasible variables wrt to x2 in batch: {total_infeas}")

            # slackz1 = (z1.squeeze(-1)+ muB)    # [B,n]
            # slackz2 = (z2.squeeze(-1)+ muB)    # [B,n]

            # # build masks of the exact indices that went non-positive
            # bad_z1 = (slackz1 <= 0).unsqueeze(-1)  # [B,n,1]
            # bad_z2 = (slackz2 <= 0).unsqueeze(-1)  # [B,n,1]

            # # Option A: out-of-place with torch.where
            # z1 = torch.where(bad_z1.squeeze(-1), torch.zeros_like(z1), z1)
            # z2 = torch.where(bad_z2.squeeze(-1), torch.zeros_like(z2), z2)
            
 
            
            # grad_x_should_be, grad_s, grad_y, grad_vv, grad_z1_should_be, grad_z2_should_be, grad_w = problem.merit_grad_M(
            #     x, x1, x2, s,       # add s1 and s2 if there are bounds for inequality
            #     y, v, torch.tensor([[[z1[0][0][0]], [2.3896]]], device=self.device), z2, w,    # add w1 and w2 instead of w if there are bounds for inequality
            #     x1E, x2E, sE, yE, vE, torch.tensor([[[z1[0][0][0]], [2.3896]]], device=self.device), z2E, wE, # add w1E and w2E instead of wE if there are bounds for inequality
            #     muP,
            #     muB, bad_x1, bad_x2, bad_z1, bad_z2
            # )

            if (self.device == 'cpu') & (print_level == 3):
                # f_val, lb_1, lb_2, lb_3, lb_4, AL_1, ub_1, ub_2, ub_3, ub_4, AL_2 = problem.merit_M_indi(x, x1, x2, s,      
                # y, v, z1, z2, w,    
                # x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                # muP, muB, bad_x1, bad_x2, bad_z1, bad_z2)

                merit_M = problem.merit_M(x, x1, x2, s,      
                y, v, z1, z2, w,    
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
                muP, muB, bad_x1, bad_x2, bad_z1, bad_z2)

                grad_x, grad_s, grad_y, grad_vv, grad_z1, grad_z2, grad_w = problem.merit_grad_M(
                x, x1, x2, s,       # add s1 and s2 if there are bounds for inequality
                y, v, z1, z2, w,    # add w1 and w2 instead of w if there are bounds for inequality
                x1E, x2E, sE, yE, vE, z1E, z2E, wE, # add w1E and w2E instead of wE if there are bounds for inequality
                muP,
                muB, bad_x1, bad_x2, bad_z1, bad_z2
                )

                # f_val_should_be, lb_1_should_be, lb_2_should_be, lb_3_should_be, lb_4_should_be, AL_1_should_be, ub_1_should_be, ub_2_should_be, ub_3_should_be, ub_4_should_be, AL_2_should_be = problem.merit_M_indi(x, x1, x2, s,      
                #     y, v, torch.tensor([[[z1[0][0][0]], [2.3896]]], device=self.device), z2, w,    
                #     x1E, x2E, sE, yE, vE, torch.tensor([[[z1[0][0][0]], [2.3896]]], device=self.device), z2E, wE, 
                #     muP, muB, bad_x1, bad_x2, bad_z1, bad_z2)

                # if grad_x is not None: 
                #     grad_x_should_be = grad_x_should_be.view(B,   n)
                # if grad_z1 is not None:
                #     grad_z1_should_be = grad_z1_should_be.view(B, n)
                # if grad_z2 is not None:
                #     grad_z2_should_be = grad_z2_should_be.view(B, n)

                if grad_x is not None: 
                    grad_x = grad_x.view(B,   n)
                if grad_z1 is not None:
                    grad_z1 = grad_z1.view(B, n)
                if grad_z2 is not None:
                    grad_z2_= grad_z2.view(B, n)

                print("x: ", x.mean().item())
                for i in range(min(5, lb.shape[0])):
                    print(f"Problem {i}:")
                    print(f"  muB {i} =  {muB[i]}")
                    print(f"  merit_M {i} =  {merit_M[i]}")
                    # print(f"  Mtest {i} =  {Mtest[i]}")
                    # print(f"  f_val {i} =  {f_val[i]}")
                    # print(f"  lb_1 {i} =  {lb_1[i]}")
                    # print(f"  lb_2 {i} =  {lb_2[i]}")
                    # print(f"  lb_3 {i} =  {lb_3[i]}")
                    # print(f"  lb_4 {i} =  {lb_4[i]}")
                    # print(f"  AL_1 {i} =  {AL_1[i]}")
                    # print(f"  ub_1 {i} =  {ub_1[i]}")
                    # print(f"  ub_2 {i} =  {ub_2[i]}")
                    # print(f"  ub_3 {i} =  {ub_3[i]}")
                    # print(f"  ub_4 {i} =  {ub_4[i]}")
                    # print(f"  AL_2 {i} =  {AL_2[i]}")
                    for j in range(lb.shape[1]):
                        print(f"  x {j} =  {x[i, j]}")
                    #     print(f"  z1 {j} =  {z1[i, j]}")
                    #     print(f"  z2 {j} =  {z2[i, j]}")
                    #     print(f"  x1 {j} =  {x1[i, j]}")
                    #     print(f"  x2 {j} =  {x2[i, j]}")
                    #     print(f"  z1E {j} =  {z1E[i, j]}")
                    #     print(f"  z2E {j} =  {z2E[i, j]}")
                    #     print(f"  x1E {j} =  {x1E[i, j]}")
                    #     print(f"  x2E {j} =  {x2E[i, j]}")
                    #     print(f"  p_x {j} =  {p_x[i, j]}")
                    #     print(f"  p_z1 {j} =  {p_z1[i, j]}")
                    #     print(f"  p_z2 {j} =  {p_z2[i, j]}")
                        print(f"  grad_x {j} =  {grad_x[i, j]}")
                    #     print(f"  grad_z1 {j} =  {grad_z1[i, j]}")
                    #     print(f"  grad_z2 {j} =  {grad_z2[i, j]}")
                        # print(f"  grad_x_should_be {j} =  {grad_x_should_be[i, j]}")
                        # print(f"  grad_z1_should_be {j} =  {grad_z1_should_be[i, j]}")
                        # print(f"  grad_z2_should_be {j} =  {grad_z2_should_be[i, j]}")
                    print()

            # Mtest = problem.Mtest(x, x1, x2, s,      
            #     y, v, z1, z2, w,    
            #     x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
            #     muP, muB, M_max, bad_x1, bad_x2, bad_z1, bad_z2)   
            
            #print("Successful number: ", (Mtest.abs() < 0.1).sum(dim = 0))

            # if (Mtest.abs() < 0.1).all():
            #     print("SUCCESS")
            #     return True  


            # # computing old merit function
            # M_proj_old = problem.merit_M(
            #     x_col, s_full,
            #     y_full, w_full,
            #     sE_flat.unsqueeze(-1),
            #     yE_flat.unsqueeze(-1),
            #     wE_flat.unsqueeze(-1),
            #     muP.view(-1,1,1),
            #     muB.view(-1,1,1)
            # ).squeeze(-1).squeeze(-1) 

            # f_val, lb_1, lb_2, lb_3, lb_4, AL_1, ub_1, ub_2, ub_3, ub_4, AL_2 = problem.merit_M_indi(x, x1, x2, s,      
            #     y, v, z1, z2, w,    
            #     x1E, x2E, sE, yE, vE, z1E, z2E, wE, 
            #     muP, muB, bad_x1, bad_x2, bad_z1, bad_z2)
            # print("f_val: ", f_val.mean().item())
            # print("lb_1: ", lb_1.mean().item())
            # print("lb_2: ", lb_2.mean().item())
            # print("lb_3: ", lb_3.mean().item())
            # print("lb_4: ", lb_4.mean().item())
            # print("AL_1: ", AL_1.mean().item())
            # print("ub_1: ", ub_1.mean().item())
            # print("ub_2: ", ub_2.mean().item())
            # print("ub_3: ", ub_3.mean().item())
            # print("ub_4: ", ub_4.mean().item())
            # print("AL_2: ", AL_2.mean().item())


            # computing your step‐loss on the projected point 
            M_proj_new = problem.merit_M(
                x, x1, x2, s,       # add s1 and s2 if there are bounds for inequality
                y, v, z1, z2, w,    # add w1 and w2 instead of w if there are bounds for inequality
                x1E.unsqueeze(-1), x2E.unsqueeze(-1), sE, yE, vE, z1E.unsqueeze(-1), z2E.unsqueeze(-1), wE, # add w1E and w2E instead of wE if there are bounds for inequality
                muP,
                muB, bad_x1, bad_x2, bad_z1, bad_z2
            )

            M_i = M_proj_new.squeeze(-1).squeeze(-1)  # [B]
            step_losses.append(M_i)

            if (self.device == 'cpu') & (print_level == 3):
                print("M_proj_new: ", M_proj_new.mean())
            #print(x.shape)

            total = total + M_proj_new



            # convert to column form for merit grad function
            # x = x.unsqueeze(-1)   
            # x1 = x1.unsqueeze(-1)   
            # x2= x2.unsqueeze(-1)
            # z1 = z1.unsqueeze(-1)
            # z2 = z2.unsqueeze(-1)

            # print("x shape at end: ", x.shape)
        
        x = x.squeeze(-1)   
        x1 = x1.squeeze(-1)   
        x2= x2.squeeze(-1)
        z1 = z1.squeeze(-1)
        z2 = z2.squeeze(-1)

        all_losses = torch.stack(step_losses, dim=0)  # [K_inner, B]
        step_loss_vector = all_losses.sum(dim=0)      # [B]

        # # --- build a per‐instance “accept” mask: only accept where M_new ≤ M_old ---
        # accept = (M_proj_new <= 1e10).float()       # [B]

        # # --- blend old vs new according to that mask ---
        # # for x (shape [B,n]), for s,y,w (shape [B,m_ineq])
        # x_out = accept.unsqueeze(1) * x_new + (1-accept).unsqueeze(1) * x_flat
        # s_out = accept.unsqueeze(1) * s_new + (1-accept).unsqueeze(1) * s_flat
        # y_out = accept.unsqueeze(1) * y_new + (1-accept).unsqueeze(1) * y_flat
        # w_out = accept.unsqueeze(1) * w_new + (1-accept).unsqueeze(1) * w_flat

        # r_out = accept.unsqueeze(1) * r_new + (1-accept).unsqueeze(1) * r_k

        # # --- define your step‐loss to be the *accepted* merit (and average over batch) ---
        # M_step = (accept * M_proj_new + (1-accept) * M_proj_old).mean()

        # finally return the blended variables, the step‐loss, and new LSTM/r states
        return (x, x1, x2, s, y, v, z1, z2, w), step_loss_vector, ((h_new, c_new), r_k)




