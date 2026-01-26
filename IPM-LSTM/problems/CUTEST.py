import os
import time
import torch
import numpy as np
import pycutest
from torch.utils.data import Dataset  # import Dataset

def pad_array(arr, target_shape):
        """
        Pads a tensor `arr` with zeros to match the `target_shape`.
        """
        # Add extra dimensions if needed.
        if arr.dim() < len(target_shape):
            for _ in range(len(target_shape) - arr.dim()):
                arr = arr.unsqueeze(-1)
        padded = torch.zeros(target_shape, dtype=arr.dtype, device=arr.device)
        slices = tuple(slice(0, s) for s in arr.shape)
        padded[slices] = arr
        return padded

import io
import contextlib

def get_sif_params(problem_name):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        pycutest.print_available_sif_params(problem_name)
    output = f.getvalue()
    return output

import re

def parse_min_param_value(sif_output, param_name):
    matches = re.findall(rf"{param_name} = (\d+)", sif_output)
    if matches:
        return int(min(matches, key=int))
    return None


class CUTEST(Dataset):
    def __init__(self, learning_type, file_path, val_frac=0, test_frac=1, device='cpu', seed=17, **kwargs):
        super().__init__()
        self.device = device
        self.seed = seed
        self.learning_type = learning_type
        self.train_frac = 1 - val_frac - test_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        torch.manual_seed(self.seed)

        # Force dimensions using provided kwargs or defaults.
        self.num_var  = kwargs.get('num_var', 100)   # Desired padded variable dimension.
        self.num_eq   = kwargs.get('num_eq', 50)      # Desired padded equality constraint dimension.
        self.num_ineq = kwargs.get('num_ineq', 50)     # Desired padded inequality constraint dimension.

        with open(file_path, 'r') as f:
            lines = f.read().strip().split('\n')
        self.problem_names = [ln.strip() for ln in lines if ln.strip()]

        # Create lists to store problem data.
        Q_list = []
        p_list = []
        G_list = []
        c_list = []
        A_list = []
        b_list = []
        lb_list = []
        ub_list = []
        self.problems = []  # List to store problem objects.


        for name in self.problem_names:
            try:
                # Step 1: Check for tunable parameters
                sif_output = get_sif_params(name)
                n_val = parse_min_param_value(sif_output, 'N')
                m_val = parse_min_param_value(sif_output, 'M')

                sifParams = {}
                if n_val is not None:
                    sifParams['N'] = n_val
                if m_val is not None:
                    sifParams['M'] = m_val

                # Step 2: Try importing with smallest parameters
                prob = pycutest.import_problem(name, sifParams=sifParams if sifParams else None)

                n = prob.n  # Actual number of decision variables.
                m = prob.m  # Total number of constraints.
                x0 = np.array(prob.x0)
                grad0 = prob.grad(x0)
                try:
                    Hess0 = prob.hess(x0)
                except Exception:
                    Hess0 = np.eye(n)
                cl = np.array(prob.cl)
                cu = np.array(prob.cu)
                eq_mask = np.isclose(cl, cu)
                ineq_mask = ~eq_mask
                num_eq_actual = np.sum(eq_mask)
                num_ineq_actual = np.sum(ineq_mask)

                if n > self.num_var or num_eq_actual > self.num_eq or num_ineq_actual > self.num_ineq:
                    print(f"[CUTEST] Skipping {name} — too large (n={n}, eq={num_eq_actual}, ineq={num_ineq_actual})")
                    continue
                self.problems.append(prob)
                Q_list.append(Hess0)
                p_list.append(grad0)
                
                # Process constraints.
                if m is None:
                    m = 0
                if m > 0:
                    cons_val = prob.cons(x0)
                    J = self.finite_diff_jac(prob.cons, x0)
                    cl = np.array(prob.cl)
                    cu = np.array(prob.cu)

                    # Identify equality and inequality indices
                    eq_mask = np.isclose(cl, cu)
                    ineq_mask = ~eq_mask

                    # Split constraints based on those masks
                    A = J[eq_mask]
                    b = cons_val[eq_mask].reshape(-1)

                    G = J[ineq_mask]
                    c = cons_val[ineq_mask].reshape(-1)

                    if A.shape[0] > 0:
                        A_list.append(A)
                        b_list.append(b)
                    else:
                        A_list.append(np.zeros((0, n)))
                        b_list.append(np.zeros(0))

                    if G.shape[0] > 0:
                        G_list.append(G)
                        c_list.append(c)
                    else:
                        G_list.append(np.zeros((0, n)))
                        c_list.append(np.zeros(0))

                
                # Process bounds.
                if hasattr(prob, 'lb') and prob.lb is not None:
                    lb = np.array(prob.lb)
                    # Replace any -infinity values with a large negative number (e.g., -1e6)
                    lb[np.isinf(lb)] = -1e6
                    lb_list.append(lb)
                else:
                    lb_list.append(-1e6 * np.ones(n))

                if hasattr(prob, 'ub') and prob.ub is not None:
                    ub = np.array(prob.ub)
                    # Replace any +infinity values with a large positive number (e.g., 1e6)
                    ub[np.isinf(ub)] = 1e6
                    ub_list.append(ub)
                else:
                    ub_list.append(1e6 * np.ones(n))

            except Exception as e:
                print(f"[CUTEST] Skipping problem {name} due to error: {e}")
                continue

        self.data_size = len(Q_list)
        if self.data_size == 0:
            raise ValueError("No valid CUTEST problems loaded!")
        
        # Save the loaded lists.
        self.Q_list = Q_list
        self.p_list = p_list
        self.G_list = G_list
        self.c_list = c_list
        self.A_list = A_list
        self.b_list = b_list
        self.lb_list = lb_list
        self.ub_list = ub_list

        # Here we use the forced padded dimensions.
        self.num_lb = self.num_var
        self.num_ub = self.num_var

        # Split indices for train/val/test.
        idx_all = torch.randperm(self.data_size, generator=torch.Generator().manual_seed(seed))
        self.train_size = int(self.data_size * (1 - self.val_frac - self.test_frac))
        self.val_size = int(self.data_size * self.val_frac)
        self.test_size = self.data_size - self.train_size - self.val_size
        if self.learning_type == 'train':
            self.indices = idx_all[:self.train_size]
        elif self.learning_type == 'val':
            self.indices = idx_all[self.train_size:self.train_size + self.val_size]
        elif self.learning_type == 'test':
            self.indices = idx_all[self.train_size + self.val_size:]
        else:
            raise ValueError("learning_type must be one of train/val/test")




    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        actual_index = self.indices[index]
        Q = torch.tensor(self.Q_list[actual_index], device=self.device, dtype=torch.float32)
        p = torch.tensor(self.p_list[actual_index], device=self.device, dtype=torch.float32)
        A = (torch.tensor(self.A_list[actual_index], device=self.device, dtype=torch.float32)
            if np.array(self.A_list[actual_index]).size > 0 else None)
        b = (torch.tensor(self.b_list[actual_index], device=self.device, dtype=torch.float32).unsqueeze(-1)
            if np.array(self.b_list[actual_index]).size > 0 else None)
        G = (torch.tensor(self.G_list[actual_index], device=self.device, dtype=torch.float32)
            if np.array(self.G_list[actual_index]).size > 0 else None)
        c = (torch.tensor(self.c_list[actual_index], device=self.device, dtype=torch.float32).unsqueeze(-1)
            if np.array(self.c_list[actual_index]).size > 0 else None)
        lb = torch.tensor(self.lb_list[actual_index], device=self.device, dtype=torch.float32)
        ub = torch.tensor(self.ub_list[actual_index], device=self.device, dtype=torch.float32)
        
        return {
            "Q": Q,
            "p": p,
            "A": A,
            "b": b,
            "G": G,
            "c": c,
            "lb": lb,
            "ub": ub,
            "n": self.num_var,
            "idx": actual_index  # new field for problem index
        }



    def finite_diff_jac(self, func, x, eps=1e-8):
        x = x.flatten()
        f0 = np.asarray(func(x)).flatten()
        m = f0.size
        n = x.size
        J = np.zeros((m, n))
        for i in range(n):
            x_eps = x.copy()
            x_eps[i] += eps
            f_eps = np.asarray(func(x_eps)).flatten()
            J[:, i] = (f_eps - f0) / eps
        return J
    
    def name(self):
        return f"CUTEST_numvar{self.num_var}_ineq{self.num_ineq}_eq{self.num_eq}"

    def obj_fn(self, x, **kwargs):
        """
        x shape: [batch, n_padded, 1]
        Returns shape: [batch]
        """
        batch_size = x.shape[0]
        out = []
        indices = kwargs["indices"]  # shape [batch]
        for i in range(batch_size):
            prob = self.problems[indices[i].item()]
            n_actual = prob.n  # The actual dimension for this problem
            # Slice out only the first n_actual entries
            x_i = x[i, :n_actual, 0].detach().cpu().numpy()
            val = prob.obj(x_i)
            out.append(val)

        return torch.tensor(out, dtype=torch.float32, device=self.device)

       


    def eq_resid(self, x, A=None, b=None):
        if A is None or b is None:
            return torch.zeros((x.size(0), 0, 1), device=self.device)
        return torch.bmm(A, x) - b.unsqueeze(-1)


    def eq_dist(self, x, **kwargs):
        return torch.abs(self.eq_resid(x, **kwargs))

    def ineq_resid(self, x, G=None, c=None):
        # If G, c are not supplied, return an empty residual
        if G is None or c is None:
            return torch.zeros((x.size(0), 0, 1), device=self.device)
        # Otherwise compute G*x - c
        return torch.bmm(G, x) - c.unsqueeze(-1)


    def ineq_dist(self, x, **kwargs):
        return torch.clamp(self.ineq_resid(x, **kwargs), min=0)

    def F0(self, x, eta, lamb, s, **kwargs):
        return torch.zeros((x.size(0), 1, 1), device=self.device)

    def cal_kkt(self, x, eta, s, lamb, zl, zu, sigma, **kwargs):
        """
        Compute the full KKT residual and its Jacobian for a CUTEST problem
        using fixed (padded) dimensions.

        Expected kwargs:
        'indices': tensor of problem indices,
        'lb': padded lower bounds [batch, target_n, 1],
        'ub': padded upper bounds [batch, target_n, 1],
        'max_n': fixed number of variables (e.g., 100),
        'max_eq': fixed number of equalities (e.g., 50),
        'max_ineq': fixed number of inequalities (e.g., 50),
        and optionally: 'A', 'b', 'G', 'c'
        """
        indices = kwargs.get('indices', None)
        if indices is None:
            raise ValueError("Must provide 'indices' in kwargs")
        batch_size = x.size(0)
        target_n = kwargs.get('max_n', self.num_var)  # e.g., 100
        m_eq     = kwargs.get('max_eq', self.num_eq)    # e.g., 50
        m_ineq   = kwargs.get('max_ineq', self.num_ineq)  # e.g., 50

        # Compute objective gradients and Hessians for each sample,
        # using the problem's actual dimension then padding to target_n.
        grad_obj_list = []
        H_obj_list = []
        for i in range(batch_size):
            idx = indices[i].item()
            prob = self.problems[idx]
            active_n = prob.n  # actual number of decision variables for this problem
            # Use only the active portion of x.
            x_active = x[i, :active_n, :].squeeze(-1)  # shape: [active_n]
            x_np = x_active.cpu().numpy()
            grad_np = prob.grad(x_np)  # shape: (active_n,)
            grad_tensor = torch.tensor(grad_np, device=self.device, dtype=torch.float32).unsqueeze(-1)
            # Pad gradient to shape (target_n, 1)
            grad_tensor = pad_array(grad_tensor, (target_n, 1))
            grad_obj_list.append(grad_tensor)
            try:
                H_np = prob.hess(x_np)  # shape: (active_n, active_n)
            except Exception:
                H_np = np.eye(active_n)
            H_tensor = torch.tensor(H_np, device=self.device, dtype=torch.float32)
            # Pad Hessian to shape (target_n, target_n)
            H_tensor = pad_array(H_tensor, (target_n, target_n))
            H_obj_list.append(H_tensor)
        grad_obj = torch.stack(grad_obj_list, dim=0)  # [batch, target_n, 1]
        H_obj    = torch.stack(H_obj_list, dim=0)       # [batch, target_n, target_n]

        # Retrieve padded lower and upper bounds (each [batch, target_n, 1])
        lb = kwargs.get('lb')
        ub = kwargs.get('ub')

        # Compute mu as a scaled average complementarity.
        mu = torch.zeros((batch_size, 1, 1), device=self.device)
        if m_ineq != 0:
            mu += sigma * ((eta * s).sum(dim=1, keepdim=True))
        if target_n != 0:
            mu += sigma * ((zl * (x - lb)).sum(dim=1, keepdim=True))
            mu += sigma * ((zu * (ub - x)).sum(dim=1, keepdim=True))
            denom = m_ineq + (2 * target_n)
        else:
            denom = m_ineq
        mu = mu / (denom + 1e-8)

        # Retrieve constraint matrices (if provided)
        A = kwargs.get('A', None)
        b = kwargs.get('b', None)
        G = kwargs.get('G', None)
        c = kwargs.get('c', None)

        # Assemble residual blocks:
        # F1: Stationarity: grad f(x) + Gᵀη + Aᵀλ - zl + zu   [shape: (batch, target_n, 1)]
        F1 = grad_obj.clone()
        if m_ineq != 0 and G is not None and eta is not None:
            F1 = F1 + torch.bmm(G.transpose(1,2), eta)
        if m_eq != 0 and A is not None and lamb is not None:
            F1 = F1 + torch.bmm(A.transpose(1,2), lamb)
        F1 = F1 - zl + zu

        # F2: Inequality feasibility: G*x - c - s   [shape: (batch, m_ineq, 1)]
        if m_ineq != 0 and G is not None and c is not None:
            F2 = torch.bmm(G, x) - c - s
        else:
            F2 = torch.zeros((batch_size, 0, 1), device=self.device)
        
        # F3: Inequality complementarity: η ∘ s - μ   [shape: (batch, m_ineq, 1)]
        if m_ineq != 0:
            F3 = eta * s - mu
        else:
            F3 = torch.zeros((batch_size, 0, 1), device=self.device)
        
        # F4: Equality feasibility: A*x - b   [shape: (batch, m_eq, 1)]
        if m_eq != 0 and A is not None and b is not None:
            F4 = torch.bmm(A, x) - b
        else:
            F4 = torch.zeros((batch_size, 0, 1), device=self.device)
        
        # F5: Lower-bound complementarity: (x - lb) ∘ zl - μ   [shape: (batch, target_n, 1)]
        if lb is not None:
            F5 = (x - lb) * zl - mu
        else:
            F5 = torch.zeros((batch_size, 0, 1), device=self.device)
        
        # F6: Upper-bound complementarity: (ub - x) ∘ zu - μ   [shape: (batch, target_n, 1)]
        if ub is not None:
            F6 = (ub - x) * zu - mu
        else:
            F6 = torch.zeros((batch_size, 0, 1), device=self.device)
        
        # Concatenate all residual blocks along dimension 1.
        F = torch.cat([F1, F2, F3, F4, F5, F6], dim=1)

        # Construct the Jacobian H of the full KKT system.
        # Total dimension = target_n + 2*m_ineq + m_eq + 2*target_n.
        total_dim = target_n + 2 * m_ineq + m_eq + 2 * target_n
        I_n = torch.eye(target_n, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        I_mineq = (torch.eye(m_ineq, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
                if m_ineq > 0 else None)
        I_meq = (torch.eye(m_eq, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
                if m_eq > 0 else None)

        # Row 1: Derivatives of F1 w.r.t. [x, η, λ, s, zl, zu]:
        row1_parts = [H_obj]
        if m_ineq != 0 and G is not None:
            row1_parts.append(G.transpose(1,2))
        else:
            row1_parts.append(torch.zeros((batch_size, target_n, m_ineq), device=self.device))
        if m_eq != 0 and A is not None:
            row1_parts.append(A.transpose(1,2))
        else:
            row1_parts.append(torch.zeros((batch_size, target_n, m_eq), device=self.device))
        if m_ineq != 0:
            row1_parts.append(torch.zeros((batch_size, target_n, m_ineq), device=self.device))
        else:
            row1_parts.append(torch.zeros((batch_size, target_n, 0), device=self.device))
        row1_parts.append(-I_n)
        row1_parts.append(I_n)
        J1 = torch.cat(row1_parts, dim=2)

        # Row 2: Derivatives of F2 = G*x - c - s.
        if m_ineq != 0 and G is not None:
            J2 = torch.cat([
                G,
                torch.zeros((batch_size, m_ineq, m_ineq), device=self.device),
                torch.zeros((batch_size, m_ineq, m_eq), device=self.device),
                -I_mineq,
                torch.zeros((batch_size, m_ineq, target_n), device=self.device),
                torch.zeros((batch_size, m_ineq, target_n), device=self.device)
            ], dim=2)
        else:
            J2 = torch.zeros((batch_size, 0, total_dim), device=self.device)

        # Row 3: Derivatives of F3 = η ∘ s - μ.
        if m_ineq != 0:
            J3 = torch.cat([
                torch.zeros((batch_size, m_ineq, target_n), device=self.device),
                torch.diag_embed(s.squeeze(-1)),
                torch.zeros((batch_size, m_ineq, m_eq), device=self.device),
                torch.diag_embed(eta.squeeze(-1)),
                torch.zeros((batch_size, m_ineq, target_n), device=self.device),
                torch.zeros((batch_size, m_ineq, target_n), device=self.device)
            ], dim=2)
        else:
            J3 = torch.zeros((batch_size, 0, total_dim), device=self.device)

        # Row 4: Derivatives of F4 = A*x - b.
        if m_eq != 0 and A is not None:
            J4 = torch.cat([
                A,
                torch.zeros((batch_size, m_eq, m_ineq), device=self.device),
                torch.zeros((batch_size, m_eq, m_eq), device=self.device),
                torch.zeros((batch_size, m_eq, m_ineq), device=self.device),
                torch.zeros((batch_size, m_eq, target_n), device=self.device),
                torch.zeros((batch_size, m_eq, target_n), device=self.device)
            ], dim=2)
        else:
            J4 = torch.zeros((batch_size, 0, total_dim), device=self.device)

        # Row 5: Derivatives of F5 = (x - lb) ∘ zl - μ.
        if lb is not None:
            J5 = torch.cat([
                torch.diag_embed(zl.squeeze(-1)),
                torch.zeros((batch_size, target_n, m_ineq), device=self.device),
                torch.zeros((batch_size, target_n, m_eq), device=self.device),
                torch.zeros((batch_size, target_n, m_ineq), device=self.device),
                torch.diag_embed((x - lb).squeeze(-1)),
                torch.zeros((batch_size, target_n, target_n), device=self.device)
            ], dim=2)
        else:
            J5 = torch.zeros((batch_size, 0, total_dim), device=self.device)

        # Row 6: Derivatives of F6 = (ub - x) ∘ zu - μ.
        if ub is not None:
            J6 = torch.cat([
                -torch.diag_embed(zu.squeeze(-1)),
                torch.zeros((batch_size, target_n, m_ineq), device=self.device),
                torch.zeros((batch_size, target_n, m_eq), device=self.device),
                torch.zeros((batch_size, target_n, m_ineq), device=self.device),
                torch.zeros((batch_size, target_n, target_n), device=self.device),
                torch.diag_embed((ub - x).squeeze(-1))
            ], dim=2)
        else:
            J6 = torch.zeros((batch_size, 0, total_dim), device=self.device)

        # Stack all rows vertically.
        H = torch.cat([J1, J2, J3, J4, J5, J6], dim=1)
        return H, F, mu




    def sub_smooth_grad(self, y, J, F, epsilon=1e-6):
        """
        Compute a smoothed subgradient from the Jacobian J and the residual F.
        This placeholder divides F elementwise by (|F| + epsilon) and multiplies by Jᵀ.
        """
        abs_F = torch.abs(F)
        smoothed_F = F / (abs_F + epsilon)
        # Here, J should have shape (batch, full_dim, full_dim)
        grad = torch.bmm(J.transpose(1, 2), smoothed_F)
        return grad
    def sub_objective(self, y, H, r):
        """
        Compute the surrogate (sub) objective for the CUTEST problem.
        
        Parameters:
          y : Tensor of shape [batch_size, full_dim, 1]
              where full_dim = num_var + 2*num_ineq + num_eq + num_lb + num_ub.
          H : Tensor of shape [batch_size, full_dim, full_dim]
          r : Tensor of shape [batch_size, full_dim, 1]
          
        The surrogate objective is defined as:
          1/2 ||H @ y - r||_2^2
        which expands to:
          1/2 * y^T H^T H y - y^T H^T r + 1/2 * r^T r.
          
        Returns:
          A tensor containing the surrogate objective for each batch element.
        """
        obj0 = 0.5 * torch.bmm(torch.bmm(y.permute(0, 2, 1), H.permute(0, 2, 1)), torch.bmm(H, y))
        obj1 = torch.bmm(torch.bmm(y.permute(0, 2, 1), H.permute(0, 2, 1)), r)
        obj2 = 0.5 * torch.bmm(r.permute(0, 2, 1), r)
        return obj0 + obj1 + obj2

