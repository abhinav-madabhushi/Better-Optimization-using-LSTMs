import torch, torch.nn as nn, torch.nn.functional as F, random

class PS_L20_LSTM(nn.Module):
    # ------------------------------------------------------------
    def __init__(self, problem, hidden_size=128, num_layers=1, K_inner = 1, device='cuda' if torch.cuda.is_available() else 'cpu', fixedmuNumber = 15, alpha=0.8): 
        super().__init__()
        try:
            torch.set_default_device(device)
        except AttributeError:
            pass
        self.K_inner = K_inner
        self.alpha = alpha

        self.fixedmuNumber = fixedmuNumber

        self.device = device

        self.px_feat_dim = 14
        self.lstm_px = nn.LSTM(
            input_size  = self.px_feat_dim, #+ (2*hidden_size),   # independent of n
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
        )
        self.px_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)   # per-token scalar step
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
            project_step, outer, print_level, rollout=True):
        
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

        def _clone_tensor(t):
            return t.clone() if t is not None else None

        def _clone_state():
            return (
                _clone_tensor(x), _clone_tensor(s), _clone_tensor(y), _clone_tensor(v),
                _clone_tensor(z1), _clone_tensor(z2), _clone_tensor(w1), _clone_tensor(w2),
                _clone_tensor(x1E), _clone_tensor(x2E), _clone_tensor(s1E), _clone_tensor(s2E),
                _clone_tensor(yE), _clone_tensor(vE), _clone_tensor(z1E), _clone_tensor(z2E),
                _clone_tensor(w1E), _clone_tensor(w2E), _clone_tensor(muP), _clone_tensor(muB),
                _clone_tensor(muA), _clone_tensor(bad_x1), _clone_tensor(bad_x2), _clone_tensor(M_max),
            )

        def _clone_lstm_state(states):
            if states is None:
                return None
            h, c = states
            return (_clone_tensor(h), _clone_tensor(c))

        def _run_rollout(init_state, lstm_x, lstm_s, r_hist, steps, warmup_no_grad):
            (x_local, s_local, y_local, v_local, z1_local, z2_local, w1_local, w2_local,
             x1E_local, x2E_local, s1E_local, s2E_local, yE_local, vE_local, z1E_local, z2E_local,
             w1E_local, w2E_local, muP_local, muB_local, muA_local, bad_x1_local, bad_x2_local, M_max_local) = init_state

            step_losses = []
            grad_losses = []
            states_x = lstm_x
            states_s = lstm_s
            r_k_local = r_hist

            for t in range(steps):
                grad_ctx = torch.no_grad() if t < warmup_no_grad else torch.enable_grad()
                # start here, need to make portfolio converge again; or maybe its just the iterations
                with grad_ctx:
                    grad_x, grad_s, grad_y, grad_vv, grad_z1, grad_z2, grad_w1, grad_w2 = problem.raw_kkt_gradients(
                        x_local, s_local,
                        y_local, v_local, z1_local, z2_local, w1_local, w2_local,
                        x1E_local, x2E_local, s1E_local, s2E_local, yE_local, vE_local, z1E_local, z2E_local, w1E_local, w2E_local,
                        muP_local, muB_local, muA_local, bad_x1_local, bad_x2_local
                    )

                    grad_x  = _trim_grad(grad_x, n)
                    grad_z1 = _trim_grad(grad_z1, n)
                    grad_z2 = _trim_grad(grad_z2, n)
                    grad_s  = _trim_grad(grad_s, m_ineq)
                    grad_y  = _trim_grad(grad_y, m_ineq)
                    grad_vv = _trim_grad(grad_vv, meq)
                    grad_w1 = _trim_grad(grad_w1, m_ineq)

                    grad_scales = []
                    for g in (grad_x, grad_s, grad_y, grad_vv, grad_z1, grad_z2, grad_w1, grad_w2):
                        if g is None:
                            continue
                        g_flat = g.reshape(g.shape[0], -1)
                        grad_scales.append(torch.norm(g_flat, dim=1).pow(2))
                    if grad_scales:
                        sum_per_sample = torch.stack(grad_scales, dim=0).sum(dim=0)  # [B]
                        grad_loss_t = torch.log(sum_per_sample + 1e-8)
                    else:
                        grad_loss_t = torch.zeros(B, device=x_local.device)

                    eps = 1e-12

                    if torch.is_tensor(muB_local):
                        muBv = muB_local.view(B, 1)
                    else:
                        muBv = torch.full((B, 1), float(muB_local), device=self.device)

                    lb = problem.lb.expand(B, n)
                    ub = problem.ub.expand(B, n)
                    has_lb = torch.isfinite(lb)
                    has_ub = torch.isfinite(ub)

                    x1_local = torch.where(has_lb, x_local - lb, torch.zeros_like(x_local))
                    x2_local = torch.where(has_ub, ub - x_local, torch.zeros_like(x_local))

                    inv_cap = 1e20

                    to2d = lambda t: (t.squeeze(-1) if (t is not None and t.dim() == 3) else t)
                    x1_local = to2d(x1_local);  x2_local = to2d(x2_local)
                    z1_local = to2d(z1_local);  z2_local = to2d(z2_local)
                    s_local = to2d(s_local)
                    w1_local = to2d(w1_local)
                    v_local = to2d(v_local) if v_local is not None else None

                    muB_bn = muB_local.view(B, 1).expand_as(x1_local)
                    muP_m = (muP_local.view(B, 1).expand_as(s_local)    if s_local  is not None else None)
                    muP_v = (muP_local.view(B, 1).expand_as(v_local)    if v_local  is not None else None)
                    muP_x = (muP_local.view(B, 1).expand_as(x_local)    if v_local  is not None else None)
                    muA_p = (muA_local.view(B, 1).expand(B, v_local.shape[1]) if v_local is not None else None)

                    invD1 = (z1_local + muB_bn) / (x1_local + muB_bn + eps)
                    invD2 = (z2_local + muB_bn) / (x2_local + muB_bn + eps)
                    invD1 = invD1.clamp_min(0.0).clamp_max(inv_cap)
                    invD2 = invD2.clamp_min(0.0).clamp_max(inv_cap)

                    if s_local is not None:
                        s_lb = getattr(problem, 's_lb', torch.zeros_like(s_local))
                        mask_s_lb = torch.isfinite(s_lb)
                        s1_ = torch.where(mask_s_lb, s_local - s_lb, torch.zeros_like(s_local))
                        muB_m = muBv.view(B, 1).expand_as(s1_)

                        if w1_local is not None:
                            Dw1_inv = (w1_local + muB_m) / (s_local + muB_m + eps)
                            Dw1_inv = torch.where(mask_s_lb, Dw1_inv, torch.zeros_like(Dw1_inv))
                            Dw1_inv = Dw1_inv.clamp_min(0.0).clamp_max(inv_cap)
                            Dw1 = (s_local + muB_m) / (w1_local + muB_m + eps)
                            Dw1 = Dw1.clamp_min(0.0).clamp_max(inv_cap)
                        else:
                            Dw1_inv = Dw1 = None
                    else:
                        Dw1_inv = Dw1 = None
                        mask_s_lb = None

                    if muP_m is not None:
                        DY  = muP_m.clamp_min(eps)
                    else:
                        DY = None

                    if muP_v is not None:
                        DV  = muP_v.clamp_min(eps)
                        DV_inv = (1.0 / DV).clamp_min(0.0).clamp_max(inv_cap)
                    else:
                        DV = DV_inv = None

                    if s_local is not None and w1_local is not None:
                        DW_combo = Dw1
                    else:
                        DW_combo = None

                    # optional tensors may be None when constraints are absent; use safe zeros with proper shape
                    zero_like_x = torch.zeros_like(x_local)
                    x1E_local = to2d(x1E_local) if x1E_local is not None else zero_like_x
                    x2E_local = to2d(x2E_local) if x2E_local is not None else zero_like_x
                    z1E_local = to2d(z1E_local) if z1E_local is not None else zero_like_x
                    z2E_local = to2d(z2E_local) if z2E_local is not None else zero_like_x
                    s1E_local = to2d(s1E_local) if s1E_local is not None else (None if m_ineq == 0 else zero_like_x)
                    w1E_local = to2d(w1E_local) if w1E_local is not None else (None if m_ineq == 0 else zero_like_x)
                    yE_local  = to2d(yE_local)  if yE_local  is not None else (None if m_ineq == 0 else zero_like_x)
                    vE_local  = to2d(vE_local)  if vE_local  is not None else zero_like_x

                    log_muB_bn = torch.log(muB_bn.clamp_min(eps)) if muB_bn is not None else None

                    if v_local is not None and hasattr(problem, 'At'):
                        Atv = torch.bmm(problem.At, v_local.unsqueeze(-1)).squeeze(-1)
                        Atgradvv = torch.bmm(problem.At, grad_vv.unsqueeze(-1)).squeeze(-1)
                        AtvE = torch.bmm(problem.At, vE_local.unsqueeze(-1)).squeeze(-1)
                        log_muP_x  = torch.log(muP_x.clamp_min(eps))
                    else:
                        Atv = torch.zeros_like(x_local)
                        Atgradvv = torch.zeros_like(x_local)
                        AtvE = torch.zeros_like(x_local)
                        log_muP_x  = torch.zeros_like(x_local)
                    
                    epsilon = 1e-4

                    feat_px = torch.stack([
                        x_local,
                        x1_local,
                        x2_local,
                        z1_local,
                        z2_local,
                        grad_x, 
                        grad_z1, 
                        grad_z2, 
                        #Atv,
                        # torch.log(torch.abs(grad_x) + 1e-4),
                        # torch.log(torch.abs(grad_z1) + 1e-4),
                        # torch.log(torch.abs(grad_z2) + 1e-4),
                        # torch.sign(grad_x), 
                        # torch.sign(grad_z1), 
                        # torch.sign(grad_z2), 
                        #Atgradvv,
                        x1E_local,
                        x2E_local,
                        z1E_local,
                        z2E_local,
                        #AtvE,
                        log_muB_bn,
                        log_muP_x,
                        #invD1,
                        #invD2
                    ], dim=-1)

                    BN   = B * n
                    inp  = feat_px.reshape(BN, self.px_feat_dim).unsqueeze(1)
                    Y, (h_new_x, c_new_x) = self.lstm_px(inp, states_x)
                    p = self.px_head(Y)
                    #p_x_1, p_x_2 = p.chunk(2, dim=-1)
                    p_x_1 = torch.abs(p.reshape(B, n))
                    #p_x_2 = torch.abs(p_x_2.reshape(B, n))

                    if s_local is not None:
                        log_muP_m  = torch.log(muP_m.clamp_min(eps))
                        feat_ps = torch.stack([
                            s_local,
                            y_local,
                            w1_local,
                            grad_s,
                            grad_y,
                            grad_w1,
                            s1E_local,
                            yE_local,
                            w1E_local,
                            log_muP_m,
                            DW_combo
                        ], dim=-1)

                        Bm   = B * m_ineq
                        inp  = feat_ps.reshape(Bm, self.ps_feat_dim).unsqueeze(1)
                        Y, (h_new_s, c_new_s) = self.lstm_ps(inp, states_s)
                        p_s = self.ps_head(Y).reshape(B, m_ineq)
                        p_s = torch.abs(p_s)
                    else:
                        (h_new_s, c_new_s) = (None, None)
                        p_s = None
                    
                    grad_x, grad_s, grad_y, grad_vv, grad_z1, grad_z2, grad_w1 = problem.merit_grad_M(
                        x_local, s_local,
                        y_local, v_local, z1_local, z2_local, w1_local, w2_local,
                        x1E_local, x2E_local, s1E_local, s2E_local, yE_local, vE_local, z1E_local, z2E_local, w1E_local, w2E_local,
                        muP_local, muB_local, muA_local, bad_x1_local, bad_x2_local
                    )

                    grad_x  = _trim_grad(grad_x, n)
                    grad_z1 = _trim_grad(grad_z1, n)
                    grad_z2 = _trim_grad(grad_z2, n)
                    grad_s  = _trim_grad(grad_s, m_ineq)
                    grad_y  = _trim_grad(grad_y, m_ineq)
                    grad_vv = _trim_grad(grad_vv, meq)
                    grad_w1 = _trim_grad(grad_w1, m_ineq)

                    if m_ineq > 0:
                        muB_m = muB_local.view(B, 1).expand(B, m_ineq)
                        s_lb = getattr(problem, 's_lb', torch.zeros_like(s_local))
                        mask_s_lb = torch.isfinite(s_lb)
                        s1_local = torch.where(mask_s_lb, s_local - s_lb, torch.zeros_like(s_local))
                        S1mu = (s1_local + muB_m).clamp_min(eps)
                        piW = muB_m * (w1E_local - s1_local + s1E_local) / S1mu
                        G = problem.G if hasattr(problem, 'G') else self.G
                        Gb = G if G.dim() == 3 else G.unsqueeze(0).expand(B, -1, -1)
                        denom = DY + DW_combo + eps
                        r_val = grad_y / denom
                        Jt_r = torch.einsum('bmn,bm->bn', Gb, r_val)
                    if meq and meq > 0:
                        A = problem.A if hasattr(problem, 'A') else self.A
                        Ab = A if A.dim() == 3 else A.unsqueeze(0).expand(B, -1, -1)

                    #dx = (- p_x_1 * torch.sign(grad_x))
                    dx = (- p_x_1 * grad_x)
                    if s_local is not None:
                        dx = dx + (- p_x_1 * Jt_r)
                    if v_local is not None:
                        dx = dx + (- p_x_1 * torch.bmm(problem.At, (DV_inv * grad_vv).unsqueeze(-1)).squeeze(-1))

                    dx1 = torch.where(has_lb,  dx,  torch.zeros_like(dx))
                    dx2 = torch.where(has_ub, -dx,  torch.zeros_like(dx))
                    X1mu = (x1_local + muB_bn).clamp_min(eps)
                    X2mu = (x2_local + muB_bn).clamp_min(eps)

                    dz1 = -( z1_local * (x1_local + dx + muB_bn) - (muB_bn * z1E_local) + muB_bn * (- x1E_local + x1_local + dx) ) / X1mu
                    dz1 = torch.where(has_lb, dz1, torch.zeros_like(dz1))

                    dz2 = -( z2_local * (x2_local - dx + muB_bn) - (muB_bn * z2E_local) + muB_bn * (- x2E_local + x2_local - dx) ) / X2mu
                    dz2 = torch.where(has_ub, dz2, torch.zeros_like(dz2))

                    if m_ineq > 0:
                        Jdx = torch.bmm(Gb, dx.unsqueeze(-1)).squeeze(-1)
                        dy = (-grad_y - Jdx) / (DY + DW_combo + eps)
                        ds = - p_s * grad_s
                        term_w1 = (w1_local * ((s_local + ds) - s_lb + muB_m)) - (muB_m * w1E_local) + (muB_m * (- s1E_local + s_local + ds))
                        dw1 = -(term_w1 / S1mu) * mask_s_lb
                    else:
                        dy = ds = dw1 = None

                    if meq and meq > 0:
                        Adx = torch.bmm(Ab, dx.unsqueeze(-1)).squeeze(-1)
                        dvv = (-grad_vv - Adx) / (DV + eps)
                        vv_new = v_local + dvv
                    else:
                        vv_new = None

                    x_new  = x_local  + dx
                    z1_new = z1_local + dz1
                    z2_new = z2_local + dz2

                    if m_ineq > 0:
                        y_new  = y_local  + dy
                        s_new  = s_local  + ds
                        w1_new = w1_local + dw1
                    else:
                        y_new, s_new, w1_new = y_local, s_local, w1_local

                    if problem.num_eq > 0:
                        vv_new = vv_new.unsqueeze(-1)
                        vE_local = vE_local.unsqueeze(-1)
                    if problem.num_ineq > 0:
                        s_local = s_local.unsqueeze(-1)
                        s_new = s_new.unsqueeze(-1)
                        s1E_local = s1E_local.unsqueeze(-1)
                        y_local = y_local.unsqueeze(-1)
                        y_new = y_new.unsqueeze(-1)
                        yE_local = yE_local.unsqueeze(-1)
                        w1_local = w1_local.unsqueeze(-1)
                        w1_new = w1_new.unsqueeze(-1)
                        w1E_local = w1E_local.unsqueeze(-1)

                    x_local, s_local, y_local, v_local, z1_local, z2_local, w1_local = project_step(
                        problem,
                        x_local.unsqueeze(-1), s_local,
                        y_local, v_local, z1_local.unsqueeze(-1), z2_local.unsqueeze(-1), w1_local, w2_local,
                        x_new.unsqueeze(-1), s_new,
                        y_new, vv_new, z1_new.unsqueeze(-1), z2_new.unsqueeze(-1), w1_new, w2_local,
                        x1E_local.unsqueeze(-1), x2E_local.unsqueeze(-1), s1E_local, s2E_local, yE_local, vE_local, z1E_local.unsqueeze(-1), z2E_local.unsqueeze(-1), w1E_local, w2E_local,
                        muP_local,
                        muB_local, muA_local
                    )

                    if problem.num_eq > 0:
                        vv_new = vv_new.squeeze(-1)
                        vE_local = vE_local.squeeze(-1)

                    if problem.num_ineq > 0:
                        s_local = s_local.squeeze(-1)
                        s_new = s_new.squeeze(-1)
                        s1E_local = s1E_local.squeeze(-1)
                        y_local = y_local.squeeze(-1)
                        y_new = y_new.squeeze(-1)
                        yE_local = yE_local.squeeze(-1)
                        w1_local = w1_local.squeeze(-1)
                        w1_new = w1_new.squeeze(-1)
                        w1E_local = w1E_local.squeeze(-1)

                    x_local   = x_local.squeeze(-1)   if x_local.dim()==3   else x_local
                    z1_local = z1_local.squeeze(-1)  if z1_local.dim()==3  else z1_local
                    z2_local = z2_local.squeeze(-1)  if z2_local.dim()==3  else z2_local
                    if problem.num_eq > 0:
                        v_local = v_local.squeeze(-1)  if v_local.dim()==3  else v_local
                    if problem.num_ineq > 0:
                        s_local   = s_local.squeeze(-1)   if s_local.dim()==3   else s_local
                        y_local   = y_local.squeeze(-1)   if y_local.dim()==3   else y_local
                        w1_local  = (w1_local.squeeze(-1) if (w1_local is not None and w1_local.dim()==3) else w1_local)

                    M_proj_new = problem.merit_M(
                        x_local, s_local,
                        y_local, v_local, z1_local, z2_local, w1_local, w2_local,
                        x1E_local, x2E_local, s1E_local, s2E_local, yE_local, vE_local, z1E_local, z2E_local, w1E_local, w2E_local,
                        muP_local,
                        muB_local, muA_local, bad_x1_local, bad_x2_local
                    )

                    M_i = M_proj_new.squeeze(-1).squeeze(-1)
                    step_losses.append(M_i)
                    grad_losses.append(grad_loss_t)

                    states_x = (h_new_x, c_new_x)
                    states_s = (h_new_s, c_new_s)

            x_final   = x_local.unsqueeze(-1)   if x_local.dim()==2   else x_local
            z1_final = z1_local.unsqueeze(-1)  if z1_local.dim()==2  else z1_local
            z2_final = z2_local.unsqueeze(-1)  if z2_local.dim()==2  else z2_local
            if problem.num_eq > 0:
                v_final = v_local.unsqueeze(-1)  if v_local.dim()==2  else v_local
            else:
                v_final = v_local
            if problem.num_ineq > 0:
                s_final   = s_local.unsqueeze(-1)   if s_local.dim()==2   else s_local
                y_final   = y_local.unsqueeze(-1)   if y_local.dim()==2   else y_local
                w1_final  = (w1_local.unsqueeze(-1) if (w1_local is not None and w1_local.dim()==2) else w1_local)
            else:
                s_final, y_final, w1_final = s_local, y_local, w1_local

            all_losses = torch.stack(step_losses, dim=0)
            all_grad_losses = torch.stack(grad_losses, dim=0) if grad_losses else torch.zeros_like(all_losses)
            step_loss_vector = all_losses.sum(dim=0)
            grad_loss_vector = all_grad_losses.sum(dim=0)
            return (
                (x_final, s_final, y_final, v_final, z1_final, z2_final, w1_final, w2_local),
                step_loss_vector,
                grad_loss_vector,
                (states_x, states_s, r_k_local),
                all_losses,
                all_grad_losses,
            )

        n_prog = random.randint(0, self.K_inner - 1) if self.K_inner > 1 else 0
        k_prog = random.randint(1, self.K_inner - n_prog) if self.K_inner - n_prog > 0 else 1

        init_state = _clone_state()
        init_lstm_x = _clone_lstm_state(states_x)
        init_lstm_s = _clone_lstm_state(states_s)
        init_rk = _clone_tensor(r_k)

        if rollout:
            # single rollout: reuse losses for both "progressive" and "full" without doubling work
            full_state, full_loss, full_grad_loss, (states_x, states_s, r_k), all_losses, all_grad_losses = _run_rollout(
                init_state, init_lstm_x, init_lstm_s, init_rk, steps=self.K_inner, warmup_no_grad=0
            )
            prog_steps = max(1, min(self.K_inner, n_prog + k_prog))
            prog_loss = all_losses[:prog_steps].sum(dim=0)
            prog_grad_loss = all_grad_losses[:prog_steps].sum(dim=0)
            blended_loss = (1 - self.alpha) * full_loss + self.alpha * prog_loss
            blended_grad_loss = (1 - self.alpha) * full_grad_loss + self.alpha * prog_grad_loss
        else:
            # single rollout only; skip progressive pass to save work
            full_state, full_loss, full_grad_loss, (states_x, states_s, r_k), all_losses_single, all_grad_losses = _run_rollout(
                init_state, init_lstm_x, init_lstm_s, init_rk, steps=self.K_inner, warmup_no_grad=0
            )
            blended_loss = full_loss
            blended_grad_loss = full_grad_loss

        return full_state, blended_loss, blended_grad_loss, (states_x, states_s, r_k)
