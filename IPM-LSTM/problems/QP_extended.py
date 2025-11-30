import os
import osqp
import time
import torch
import numpy as np
import scipy.io as sio
import cyipopt as ipopt

from scipy.sparse import csc_matrix

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class QP(object):
    """
        minimize_x 0.5*x^T Q x + p^Tx
        s.t.       Gx <= c
                   Ax = b

        Q: [batch_size, num_var, num_var]
        p: [batch_size, num_var, 1]
        G: [batch_size, num_ineq, num_var]
        c: [batch_size, num_ineq, 1]
        A: [batch_size, num_eq, num_var]
        b: [batch_size, num_eq, 1]
    """
    def __init__(self, prob_type, learning_type, val_frac=0.0001, test_frac=0.9000, device='cuda' if torch.cuda.is_available() else 'cpu', seed=17, **kwargs):
        super().__init__()

        self.device = device
        self.seed = seed
        self.learning_type = learning_type
        self.train_frac = 1 - val_frac - test_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.prob_type = prob_type
        torch.manual_seed(self.seed)

        if prob_type == 'QP_RHS':
            file_path = kwargs['file_path']
            data = sio.loadmat(file_path)
            self.data_size = data['b'].shape[0]
            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size

            self.num_var = data['Q'].shape[1]
            self.num_ineq = data['G'].shape[1]
            self.num_eq = data['A'].shape[1]
            self.num_lb = 0
            self.num_ub = 0
                        
            if learning_type == 'train':
                self.Q = torch.tensor(data['Q'][:self.train_size], device=self.device).float()  # (train_size, num_var, num_var)
                self.p = torch.tensor(data['p'][:self.train_size], device=self.device).float().unsqueeze(-1)  # (train_size, num_var, 1)
                self.A = torch.tensor(data['A'][:self.train_size], device=self.device).float()  # (train_size, num_eq, num_var)
                self.b = torch.tensor(data['b'][:self.train_size], device=self.device).float()  # (train_size, num_eq, 1)
                self.G = torch.tensor(data['G'][:self.train_size], device=self.device).float()  # (train_size, num_ineq, num_var)
                self.c = torch.tensor(data['c'], device=self.device).float().expand(self.train_size, -1, -1)  # (train_size, num_ineq, 1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'val':
                self.Q = torch.tensor(data['Q'][:self.val_size], device=self.device).float()  # (train_size, num_var, num_var)
                self.p = torch.tensor(data['p'][:self.val_size], device=self.device).float().unsqueeze(-1)  # (train_size, num_var, 1)
                self.A = torch.tensor(data['A'][:self.val_size], device=self.device).float()  # (train_size, num_eq, num_var)
                self.b = torch.tensor(data['b'][:self.val_size], device=self.device).float()  # (train_size, num_eq, 1)
                self.G = torch.tensor(data['G'][:self.val_size], device=self.device).float()  # (train_size, num_ineq, num_var)
                self.c = torch.tensor(data['c'], device=self.device).float().expand(self.val_size, -1, -1)  # (train_size, num_ineq, 1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'test':
                self.Q = torch.tensor(data['Q'][:self.test_size], device=self.device).float()  # (train_size, num_var, num_var)
                self.p = torch.tensor(data['p'][:self.test_size], device=self.device).float().unsqueeze(-1)  # (train_size, num_var, 1)
                self.A = torch.tensor(data['A'][:self.test_size], device=self.device).float()  # (train_size, num_eq, num_var)
                self.b = torch.tensor(data['b'][:self.test_size], device=self.device).float()  # (train_size, num_eq, 1)
                self.G = torch.tensor(data['G'][:self.test_size], device=self.device).float()  # (train_size, num_ineq, num_var)
                self.c = torch.tensor(data['c'], device=self.device).float().expand(self.test_size, -1, -1) # (train_size, num_ineq, 1)
                self.lb = -torch.inf
                self.ub = torch.inf

        elif prob_type == 'QP':
            self.data_size = kwargs['data_size']
            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size

            self.num_var = kwargs['num_var']
            self.num_ineq = kwargs['num_ineq']
            self.num_eq = kwargs['num_eq']
            self.num_lb = 0
            self.num_ub = 0

            if learning_type == 'train':
                self.Q = torch.diag_embed(torch.rand(size=(self.data_size, self.num_var), device=device))[:self.train_size]
                self.p = torch.rand(size=(self.data_size, self.num_var), device=device)[:self.train_size].unsqueeze(-1)
                self.A = torch.normal(mean=0, std=1, size=(self.data_size, self.num_eq, self.num_var), device=device)[:self.train_size]
                self.b = 2 * torch.rand(size=(self.data_size, self.num_eq), device=device)[:self.train_size].unsqueeze(-1) - 1  # [-1, 1]
                self.G = torch.normal(mean=0, std=1, size=(self.data_size, self.num_ineq, self.num_var), device=device)[:self.train_size]
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'val':
                self.Q = torch.diag_embed(torch.rand(size=(self.data_size, self.num_var), device=device))[self.train_size:self.train_size + self.val_size]
                self.p = torch.rand(size=(self.data_size, self.num_var), device=device)[self.train_size:self.train_size + self.val_size].unsqueeze(-1)
                self.A = torch.normal(mean=0, std=1, size=(self.data_size, self.num_eq, self.num_var), device=device)[self.train_size:self.train_size + self.val_size]
                self.b = 2 * torch.rand(size=(self.data_size, self.num_eq), device=device)[self.train_size:self.train_size + self.val_size].unsqueeze(-1) - 1  # [-1, 1]
                self.G = torch.normal(mean=0, std=1, size=(self.data_size, self.num_ineq, self.num_var), device=device)[self.train_size:self.train_size + self.val_size]
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'test':
                self.Q = torch.diag_embed(torch.rand(size=(self.data_size, self.num_var), device=device))[self.train_size + self.val_size:]
                self.p = torch.rand(size=(self.data_size, self.num_var), device=device)[self.train_size + self.val_size:].unsqueeze(-1)
                self.A = torch.normal(mean=0, std=1, size=(self.data_size, self.num_eq, self.num_var), device=device)[self.train_size + self.val_size:]
                self.b = 2 * torch.rand(size=(self.data_size, self.num_eq), device=device)[self.train_size + self.val_size:].unsqueeze(-1) - 1  # [-1, 1]
                self.G = torch.normal(mean=0, std=1, size=(self.data_size, self.num_ineq, self.num_var), device=device)[self.train_size + self.val_size:]
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf

        else:
            file_path = kwargs['file_path']
            data = sio.loadmat(file_path)
            self.data_size = data['Q'].shape[0]
            self.n_all = (
                torch.as_tensor(np.asarray(data['n']).squeeze(), device=self.device).long()
                if 'n' in data else None
            )

            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size

            #print(self.test_size)

            self.num_var = data['Q'].shape[1]
            try:
                self.num_ineq = data['G'].shape[1]
            except KeyError:
                self.num_ineq = 0
            
            # try:
            #     self.num_ineq_lb = data['c1'].shape[1]
            # except KeyError:
            #     self.num_ineq_lb = 0

            # try:
            #     self.num_ineq_lb = data['c2'].shape[1]
            # except KeyError:
            #     self.num_ineq_lb = 0

            try:
                self.num_eq = data['A'].shape[1]
            except KeyError:
                self.num_eq = 0

            try:
                self.num_lb = data['lb'].shape[1]
            except KeyError:
                self.num_lb = 0

            try:
                self.num_ub = data['ub'].shape[1]
            except KeyError:
                self.num_ub = 0

            if learning_type == 'train':
                self.n_vec = None if self.n_all is None else self.n_all[:self.train_size]
                self.Q = torch.tensor(data['Q'], device=self.device).float()[:self.train_size]
                self.p = torch.tensor(data['p'].astype(np.float32), device=self.device).float()[:self.train_size].unsqueeze(-1)
                self.pt = self.p.permute(0, 2, 1)
                self.Q_sym = 0.5*(self.Q+self.Q.permute(0,2,1))
                B, n = self.Q.shape[0], self.Q.shape[1]
                if self.num_eq != 0:
                    self.A = torch.tensor(data['A'], device=self.device).float()[:self.train_size]
                    #print(self.A.shape)
                    self.At = self.A.transpose(1, 2).contiguous()
                    self.b = torch.tensor(data['b'].astype(np.float32), device=self.device).float()[:self.train_size]
                if self.num_ineq != 0:
                    self.G = torch.tensor(data['G'], device=self.device).float()[:self.train_size]
                    self.c = torch.tensor(data['c'].astype(np.float32), device=self.device).float()[:self.train_size]
                if self.num_lb != 0:
                    self.lb = torch.tensor(data['lb'], device=self.device).float()[:self.train_size]
                else:
                    self.lb = -torch.inf
                if self.num_ub != 0:
                    self.ub = torch.tensor(data['ub'], device=self.device).float()[:self.train_size]
                else:
                    self.ub = torch.inf
            elif learning_type == 'val':
                self.n_vec = None if self.n_all is None else self.n_all[self.train_size:self.train_size + self.val_size]
                self.Q = torch.tensor(data['Q'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                self.p = torch.tensor(data['p'].astype(np.float32), device=self.device).float()[self.train_size:self.train_size + self.val_size].unsqueeze(-1)
                self.pt = self.p.permute(0, 2, 1)
                self.Q_sym = 0.5*(self.Q+self.Q.permute(0,2,1))
                B, n = self.Q.shape[0], self.Q.shape[1]
                if self.num_eq != 0:
                    self.A = torch.tensor(data['A'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                    self.At = self.A.transpose(1, 2).contiguous()
                    self.b = torch.tensor(data['b'].astype(np.float32), device=self.device).float()[self.train_size:self.train_size + self.val_size]
                if self.num_ineq != 0:
                    self.G = torch.tensor(data['G'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                    self.c = torch.tensor(data['c'].astype(np.float32), device=self.device).float()[self.train_size:self.train_size + self.val_size]
                if self.num_lb != 0:
                    self.lb = torch.tensor(data['lb'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                else:
                    self.lb = -torch.inf
                if self.num_ub != 0:
                    self.ub = torch.tensor(data['ub'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                else:
                    self.ub = torch.inf
            elif learning_type == 'test':
                self.n_vec = None if self.n_all is None else self.n_all[self.train_size + self.val_size:]
                self.Q = torch.tensor(data['Q'], device=self.device).float()[self.train_size + self.val_size:]
                self.p = torch.tensor(data['p'].astype(np.float32), device=self.device).float()[self.train_size + self.val_size:].unsqueeze(-1)
                self.pt = self.p.permute(0, 2, 1)
                self.Q_sym = 0.5*(self.Q+self.Q.permute(0,2,1))
                B, n = self.Q.shape[0], self.Q.shape[1]
                if self.num_eq != 0:
                    self.A = torch.tensor(data['A'], device=self.device).float()[self.train_size + self.val_size:]
                    self.At = self.A.transpose(1, 2).contiguous()
                    self.b = torch.tensor(data['b'].astype(np.float32), device=self.device).float()[self.train_size + self.val_size:]
                if self.num_ineq != 0:
                    self.G = torch.tensor(data['G'], device=self.device).float()[self.train_size + self.val_size:]
                    self.c = torch.tensor(data['c'].astype(np.float32), device=self.device).float()[self.train_size + self.val_size:]
                if self.num_lb != 0:
                    self.lb = torch.tensor(data['lb'], device=self.device).float()[self.train_size + self.val_size:]
                else:
                    self.lb = -torch.inf
                if self.num_ub != 0:
                    self.ub = torch.tensor(data['ub'], device=self.device).float()[self.train_size + self.val_size:]
                else:
                    self.ub = torch.inf


    def name(self):
        str = '{}_{}_{}_{}_{}'.format(self.prob_type, self.num_ineq, self.num_eq, self.num_lb, self.num_ub)
        return str
    
    # def expand_all_matrices(self, target_batch):
    #     def _expand_if_needed(t, target_batch):
    #         t = t.repeat_interleave(target_batch, dim=0)
    #         return t
    #     self.Q = _expand_if_needed(self.Q, target_batch)
    #     self.p = _expand_if_needed(self.p, target_batch)
    #     self.lb = _expand_if_needed(self.lb, target_batch)
    #     self.ub = _expand_if_needed(self.ub, target_batch)



    def obj_fn(self, x, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        return 0.5 * torch.bmm(x.permute(0, 2, 1), torch.bmm(Q, x)) + torch.bmm(p.permute(0, 2, 1), x)

    def obj_grad(self, x, **kwargs):
        p = kwargs.get('p', self.p)
        if 'Q' in kwargs:
            Q = kwargs['Q']
            Q_sym = 0.5 * (Q + Q.permute(0, 2, 1))
        else:
            # reuse cached symmetric copy when available
            Q_sym = getattr(self, 'Q_sym', None)
            if Q_sym is None:
                Q = self.Q
                Q_sym = 0.5 * (Q + Q.permute(0, 2, 1))
                self.Q_sym = Q_sym
        return torch.bmm(Q_sym, x) + p

    def ineq_resid(self, x, **kwargs):
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        return torch.bmm(G, x) - c

    def ineq_dist(self, x, **kwargs):
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        return torch.clamp(self.ineq_resid(x, G=G, c=c), 0)

    def eq_resid(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        return torch.bmm(A, x) - b

    def eq_dist(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        return torch.abs(self.eq_resid(x, A=A, b=b))

    def lower_bound_dist(self, x, **kwargs):
        lb = kwargs.get('lb', self.lb)
        return torch.clamp(lb - x, 0)

    def upper_bound_dist(self, x, **kwargs):
        ub = kwargs.get('ub', self.ub)
        return torch.clamp(x-ub, 0)
    
    def ineq_grad(self, x, **kwargs):
        # x: [B, n, 1] or [B, n]
        G = kwargs.get('G', self.G)                 # [B, m_ineq, n] or [m_ineq, n]
        if G.dim() == 2:
            B = x.shape[0]
            G = G.unsqueeze(0).expand(B, -1, -1)    # -> [B, m_ineq, n]
        return G 

    def F0(self, x, eta, s, lamb, zl, zu, sigma, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        if self.num_ineq != 0:
            G = kwargs.get('G', self.G)
            c = kwargs.get('c', self.c)
        if self.num_eq != 0:
            A = kwargs.get('A', self.A)
            b = kwargs.get('b', self.b)
        if self.num_lb != 0:
            lb = kwargs.get('lb', self.lb)
        if self.num_ub != 0:
            ub = kwargs.get('ub', self.ub)

        # residual
        F_list = []
        F1 = torch.bmm(0.5 * (Q + Q.permute(0, 2, 1)), x) + p
        if self.num_ineq != 0:
            F1 += torch.bmm(G.permute(0, 2, 1), eta)
        if self.num_eq != 0:
            F1 += torch.bmm(A.permute(0, 2, 1), lamb)
        if self.num_lb != 0:
            F1 += -zl
        if self.num_ub != 0:
            F1 += zu
        F_list.append(F1)

        if self.num_ineq != 0:
            F2 = torch.bmm(G, x) - c + s
            F3 = eta * s
            F_list.append(F2)
            F_list.append(F3)

        if self.num_eq != 0:
            F4 = torch.bmm(A, x) - b
            F_list.append(F4)

        if self.num_lb != 0:
            F5 = zl * (x - lb)
            F_list.append(F5)

        if self.num_ub != 0:
            F6 = zu * (ub - x)
            F_list.append(F6)

        F = torch.concat(F_list, dim=1)
        return F


    def cal_kkt(self, x, eta, s, lamb, zl, zu, sigma, **kwargs):
        """
        x: [batch_size, num_var, 1]
        eta: [batch_size, num_ineq, 1]
        lamb: [batch_size, num_eq, 1]
        s: [batch_size, num_ineq, 1]
        zl: [batch_size, num_lb, 1]
        zu: [batch_size, num_ub, 1]

        return:
        J: [batch_size, num_var+num_ineq+num_ineq+num_eq, num_var+num_ineq+num_ineq+num_eq]
        F: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        mu: [batch_size, 1, 1]
        """
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        mu = 0
        if self.num_ineq != 0:
            G = kwargs.get('G', self.G)
            c = kwargs.get('c', self.c)
            mu += sigma * ((eta * s).sum(1).unsqueeze(-1))
        if self.num_eq != 0:
            A = kwargs.get('A', self.A)
            b = kwargs.get('b', self.b)
        if self.num_lb != 0:
            lb = kwargs.get('lb', self.lb)
            mu += sigma * ((zl * (x-lb)).sum(1).unsqueeze(-1))
        if self.num_ub != 0:
            ub = kwargs.get('ub', self.ub)
            mu += sigma * ((zu * (ub-x)).sum(1).unsqueeze(-1))
        batch_size = Q.shape[0]
        # mu
        mu = mu/(self.num_ineq+self.num_lb+self.num_ub)

        # residual
        F_list = []
        F1 = torch.bmm(0.5*(Q+Q.permute(0,2,1)), x) + p
        if self.num_ineq != 0:
            F1 += torch.bmm(G.permute(0, 2, 1), eta)
        if self.num_eq != 0:
            F1 += torch.bmm(A.permute(0, 2, 1), lamb)
        if self.num_lb != 0:
            F1 += -zl
        if self.num_ub != 0:
            F1 += zu
        F_list.append(F1)

        if self.num_ineq != 0:
            F2 = torch.bmm(G, x) - c + s
            F3 = eta * s - mu
            F_list.append(F2)
            F_list.append(F3)

        if self.num_eq != 0:
            F4 = torch.bmm(A, x) - b
            F_list.append(F4)

        if self.num_lb != 0:
            F5 = zl*(x-lb) - mu
            F_list.append(F5)

        if self.num_ub != 0:
            F6 = zu*(ub-x) - mu
            F_list.append(F6)

        F = torch.concat(F_list, dim=1)

        # jacobian of residual
        J_list = []
        J1 = 0.5*(Q+Q.permute(0,2,1))
        if self.num_ineq != 0:
            J1 = torch.concat((J1, G.permute(0,2,1)), dim=2)
        if self.num_eq != 0:
            J1 = torch.concat((J1, A.permute(0,2,1)), dim=2)
        if self.num_ineq != 0:
            J1 = torch.concat((J1, torch.zeros(size=(batch_size, self.num_var, self.num_ineq), device=self.device)), dim=2)
        if self.num_lb != 0:
            J1 = torch.concat((J1, -torch.diag_embed(torch.ones(size=(batch_size, self.num_lb), device=self.device))), dim=2)
        if self.num_ub != 0:
            J1 = torch.concat((J1, torch.diag_embed(torch.ones(size=(batch_size, self.num_ub), device=self.device))), dim=2)
        J_list.append(J1)

        if self.num_ineq != 0:
            J2 = torch.concat((G, torch.zeros(size=(batch_size, self.num_ineq, self.num_ineq), device=self.device)), dim=2)
            if self.num_eq != 0:
                J2 = torch.concat((J2, torch.zeros(size=(batch_size, self.num_ineq, self.num_eq), device=self.device)), dim=2)
            J2 = torch.concat((J2, torch.diag_embed(torch.ones(size=(batch_size, self.num_ineq), device=self.device))), dim=2)
            if self.num_lb != 0:
                J2 = torch.concat((J2, torch.zeros(size=(batch_size, self.num_ineq, self.num_lb), device=self.device)), dim=2)
            if self.num_ub != 0:
                J2 = torch.concat((J2, torch.zeros(size=(batch_size, self.num_ineq, self.num_ub), device=self.device)), dim=2)
            J_list.append(J2)

            J3 = torch.zeros(size=(batch_size, self.num_ineq, self.num_var), device=self.device)
            J3 = torch.concat((J3, torch.diag_embed(s.squeeze(-1))), dim=2)
            if self.num_eq != 0:
                J3 = torch.concat((J3, torch.zeros(size=(batch_size, self.num_ineq, self.num_eq), device=self.device)), dim=2)
            J3 = torch.concat((J3, torch.diag_embed(eta.squeeze(-1))), dim=2)
            if self.num_lb != 0:
                J3 = torch.concat((J3, torch.zeros(size=(batch_size, self.num_ineq, self.num_lb), device=self.device)), dim=2)
            if self.num_ub != 0:
                J3 = torch.concat((J3, torch.zeros(size=(batch_size, self.num_ineq, self.num_ub), device=self.device)), dim=2)
            J_list.append(J3)

        if self.num_eq != 0:
            J4 = A
            if self.num_ineq != 0:
                J4 = torch.concat((J4, torch.zeros(size=(batch_size, self.num_eq, self.num_ineq), device=self.device)), dim=2)
            J4 = torch.concat((J4, torch.zeros(size=(batch_size, self.num_eq, self.num_eq), device=self.device)), dim=2)
            if self.num_ineq != 0:
                J4 = torch.concat((J4, torch.zeros(size=(batch_size, self.num_eq, self.num_ineq), device=self.device)), dim=2)
            if self.num_lb != 0:
                J4 = torch.concat((J4, torch.zeros(size=(batch_size, self.num_eq, self.num_lb), device=self.device)), dim=2)
            if self.num_ub != 0:
                J4 = torch.concat((J4, torch.zeros(size=(batch_size, self.num_eq, self.num_ub), device=self.device)), dim=2)
            J_list.append(J4)

        if self.num_lb != 0:
            J5 = torch.diag_embed(zl.squeeze(-1))
            if self.num_ineq != 0:
                J5 = torch.concat((J5, torch.zeros(size=(batch_size, self.num_lb, self.num_ineq), device=self.device)), dim=2)
            if self.num_eq != 0:
                J5 = torch.concat((J5, torch.zeros(size=(batch_size, self.num_lb, self.num_eq), device=self.device)), dim=2)
            if self.num_ineq != 0:
                J5 = torch.concat((J5, torch.zeros(size=(batch_size, self.num_lb, self.num_ineq), device=self.device)), dim=2)
            J5 = torch.concat((J5, torch.diag_embed((x-lb).squeeze(-1))), dim=2)
            if self.num_ub != 0:
                J5 = torch.concat((J5, torch.zeros(size=(batch_size, self.num_lb, self.num_ub), device=self.device)), dim=2)
            J_list.append(J5)

        if self.num_ub != 0:
            J6 = -torch.diag_embed(zu.squeeze(-1))
            if self.num_ineq != 0:
                J6 = torch.concat((J6, torch.zeros(size=(batch_size, self.num_ub, self.num_ineq), device=self.device)), dim=2)
            if self.num_eq != 0:
                J6 = torch.concat((J6, torch.zeros(size=(batch_size, self.num_ub, self.num_eq), device=self.device)), dim=2)
            if self.num_ineq != 0:
                J6 = torch.concat((J6, torch.zeros(size=(batch_size, self.num_ub, self.num_ineq), device=self.device)), dim=2)
            if self.num_lb != 0:
                J6 = torch.concat((J6, torch.zeros(size=(batch_size, self.num_ub, self.num_lb), device=self.device)), dim=2)
            J6 = torch.concat((J6, torch.diag_embed((ub-x).squeeze(-1))), dim=2)
            J_list.append(J6)

        J = torch.concat(J_list, dim=1)
        return J, F, mu

    def sub_objective(self, y, J, F):
        """
        y: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        J: [batch_size, num_var+num_ineq+num_ineq+num_eq, num_var+num_ineq+num_ineq+num_eq]
        F: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        1/2||J@y-F||_2^2 = 1/2(y^T@J^T@Jy)-y^TJ^TF+1/2(F^TF)
        """
        obj0 = 0.5 * torch.bmm(torch.bmm(y.permute(0, 2, 1), J.permute(0, 2, 1)), torch.bmm(J, y))
        obj1 = torch.bmm(torch.bmm(y.permute(0, 2, 1), J.permute(0, 2, 1)), F)
        obj2 = 0.5 * (torch.bmm(F.permute(0, 2, 1), F))
        return obj0+obj1+obj2

    def sub_smooth_grad(self, y, J, F):
        """
        y: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        return H^T@H@delta_r+H^T@r
        """
        grad = torch.bmm(torch.bmm(J.permute(0, 2, 1), J), y) + torch.bmm(J.permute(0, 2, 1), F)
        return grad

    def opt_solve(self, solver_type='ipopt_box_qp_extended', tol=1e-2, initial_y = None, init_mu=None, init_g=None, init_zl=None, init_zu=None):
        if solver_type == 'osqp':
            print('running osqp')
            Q, p = self.Q.detach().cpu().numpy(), self.p.detach().cpu().numpy()
            if self.num_ineq != 0:
                G, c = self.G.detach().cpu().numpy(), self.c.detach().cpu().numpy()
            if self.num_eq != 0:
                A, b = self.A.detach().cpu().numpy(), self.b.detach().cpu().numpy()
            if self.num_lb != 0:
                lb = self.lb.detach().cpu().numpy()
            if self.num_ub != 0:
                ub = self.ub.detach().cpu().numpy()

            s = []
            iters = 0
            total_time = 0
            for i in range(Q.shape[0]):
                solver = osqp.OSQP()
                A0 = []
                zl = []
                zu = []
                if self.num_ineq != 0:
                    A0.append(G[i, :, :])
                    zl.append(-np.ones(c.shape[1]) * np.inf)
                    zu.append(c[i, :])
                if self.num_eq != 0:
                    A0.append(A[i, :, :])
                    zl.append(b[i, :])
                    zu.append(b[i, :])
                if self.num_lb != 0:
                    A0.append(np.eye(p.shape[1]))
                    zl.append(lb[i, :])
                else:
                    A0.append(np.eye(p.shape[1]))
                    zl.append(-np.ones(p.shape[1])*np.inf)
                if self.num_ub != 0:
                    zu.append(ub[i, :])
                else:
                    zu.append(ub[i, :])

                my_A = np.vstack(A0)
                my_l = np.hstack(zl)
                my_u = np.hstack(zu)
                solver.setup(P=csc_matrix(Q[i, :, :]), q=p[i, :], A=csc_matrix(my_A),
                             l=my_l, u=my_u, verbose=False, eps_prim_inf=tol)
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += (end_time - start_time)
                if results.info.status == 'solved':
                    s.append(results.x)
                else:
                    s.append(np.ones(self.num_var) * np.nan)
                    print('Batch {} optimization failed.'.format(i))

            sols = np.array(s)
            parallel_time = total_time / Q.shape[0]

        elif solver_type == 'ipopt':
            Q, p = self.Q.detach().cpu().numpy(), self.p.detach().cpu().numpy()
            if self.num_ineq != 0:
                G, c = self.G.detach().cpu().numpy(), self.c.detach().cpu().numpy()
            if self.num_eq != 0:
                A, b = self.A.detach().cpu().numpy(), self.b.detach().cpu().numpy()
            if self.num_lb != 0:
                lb = self.lb.detach().cpu().numpy()
            else:
                lb = -np.infty * np.ones(shape=(Q.shape[0], Q.shape[1], 1))
            if self.num_ub != 0:
                ub = self.ub.detach().cpu().numpy()
            else:
                ub = np.infty * np.ones(shape=(Q.shape[0], Q.shape[1], 1))

            Y = []
            iters = []
            total_time = 0
            for i in range(Q.shape[0]):
                if initial_y is None:
                    # y0 = np.linalg.pinv(A[i]) @ b[i]  # feasible initial point
                    if (self.num_lb != 0) and (self.num_ub != 0):
                        y0 = ((lb[i]+ub[i])/2).squeeze(-1)
                    elif (self.num_lb != 0) and (self.num_ub == 0):
                        y0 = (lb[i] + np.ones(shape=lb[i].shape)).squeeze(-1)
                    elif (self.num_lb == 0) and (self.num_lb != 0):
                        y0 = (ub[i] - np.ones(shape=ub[i].shape)).squeeze(-1)
                    else:
                        y0 = np.zeros(self.num_var)
                else:
                    y0 = initial_y[i].cpu().numpy()

                # upper and lower bounds on constraints
                cls = []
                cus = []
                if self.num_ineq != 0:
                    cls.append(-np.inf * np.ones(G[i].shape[0]))
                    cus.append(c[i].squeeze(-1))
                if self.num_eq != 0:
                    cls.append(b[i].squeeze(-1))
                    cus.append(b[i].squeeze(-1))
                if (self.num_ineq == 0) and (self.num_eq == 0):
                    cl = []
                    cu = []
                else:
                    cl = np.hstack(cls)
                    cu = np.hstack(cus)

                if (self.num_ineq != 0) and (self.num_eq != 0):
                    G0, A0 = G[i], A[i]
                elif (self.num_ineq != 0) and (self.num_eq == 0):
                    G0, A0 = G[i], np.array(0.0)
                elif (self.num_ineq == 0) and (self.num_eq != 0):
                    G0, A0 = np.array(0.0), A[i]
                else:
                    G0, A0 = np.array(0.0), np.array(0.0)

                nlp = convex_ipopt(
                    Q[i],
                    p[i].squeeze(-1),
                    G0,
                    A0,
                    n=len(y0),
                    m=len(cl),
                    lb=lb[i],
                    ub=ub[i],
                    cl=cl,
                    cu=cu
                )

                nlp.add_option('tol', tol)
                nlp.add_option('dual_inf_tol', 1e-2)
                nlp.add_option('constr_viol_tol', 1e-2)
                nlp.add_option('compl_inf_tol', 1e-2)
                nlp.add_option('print_level', 0)  # 3)
                if init_mu is not None:
                    nlp.add_option('warm_start_init_point', 'yes')
                    nlp.add_option('warm_start_bound_push', 1e-20)
                    nlp.add_option('warm_start_bound_frac', 1e-20)
                    nlp.add_option('warm_start_slack_bound_push', 1e-20)
                    nlp.add_option('warm_start_slack_bound_frac', 1e-20)
                    nlp.add_option('warm_start_mult_bound_push', 1e-20)
                    nlp.add_option('mu_strategy', 'monotone')
                    nlp.add_option('mu_init', init_mu[i].squeeze().cpu().item())

                start_time = time.time()
                if init_g is not None:
                    g = [x.item() for x in init_g[i].cpu()]
                else:
                    g = []

                if init_zl is not None:
                    zl = [x.item() for x in init_zl[i].cpu()]
                else:
                    zl = []

                if init_zu is not None:
                    zu = [x.item() for x in init_zu[i].cpu()]
                else:
                    zu = []

                y, info = nlp.solve(y0, lagrange=g, zl=zl, zu=zu)

                end_time = time.time()
                Y.append(y)
                iters.append(len(nlp.objectives))
                total_time += (end_time - start_time)

            sols = np.array(Y)
            parallel_time = total_time / Q.shape[0]
        elif solver_type == 'ipopt_box_qp_extended':
            Q, p = self.Q.detach().cpu().numpy(), self.p.detach().cpu().numpy()

            # Optional inequality and equality matrices + RHS
            if self.num_ineq != 0:
                G = self.G.detach().cpu().numpy()   # [B, m_ineq, n]
                c = self.c.detach().cpu().numpy()   # [B, m_ineq, 1]
            else:
                G = None
                c = None

            if self.num_eq != 0:
                A = self.A.detach().cpu().numpy()   # [B, m_eq, n]
                b = self.b.detach().cpu().numpy()   # [B, m_eq, 1]
            else:
                A = None
                b = None

            # Box bounds on x
            if self.num_lb != 0:
                lb = self.lb.detach().cpu().numpy()       # [B, n, 1] or [B, n]
            else:
                lb = -np.infty * np.ones(shape=(Q.shape[0], Q.shape[1], 1))

            if self.num_ub != 0:
                ub = self.ub.detach().cpu().numpy()       # [B, n, 1] or [B, n]
            else:
                ub = np.infty * np.ones(shape=(Q.shape[0], Q.shape[1], 1))

            Y = []
            iters = []
            total_time = 0.0

            for i in range(Q.shape[0]):
                # ---- initial point y0 ----
                if initial_y is None:
                    lb_i = lb[i]  # (n,) or (n,1)
                    ub_i = ub[i]

                    # make them 1D if needed
                    if lb_i.ndim == 2 and lb_i.shape[1] == 1:
                        lb_i = lb_i.squeeze(-1)
                    if ub_i.ndim == 2 and ub_i.shape[1] == 1:
                        ub_i = ub_i.squeeze(-1)

                    n_i = lb_i.shape[0]
                    y0 = np.zeros(n_i, dtype=float)

                    finite_lb = np.isfinite(lb_i)
                    finite_ub = np.isfinite(ub_i)

                    both = finite_lb & finite_ub
                    y0[both] = 0.5 * (lb_i[both] + ub_i[both])

                    lb_only = finite_lb & ~finite_ub
                    y0[lb_only] = lb_i[lb_only] + 1.0

                    ub_only = ~finite_lb & finite_ub
                    y0[ub_only] = ub_i[ub_only] - 1.0

                    # if both infinite, leave as 0
                else:
                    y0 = initial_y[i].cpu().numpy()

                # ---- pick G0, c0, A0, b0 for this sample ----
                if (self.num_ineq != 0) and (self.num_eq != 0):
                    G0  = G[i]
                    c0  = c[i].squeeze(-1)  # (m_ineq,)
                    A0  = A[i]
                    b0  = b[i].squeeze(-1)  # (m_eq,)
                elif (self.num_ineq != 0) and (self.num_eq == 0):
                    G0  = G[i]
                    c0  = c[i].squeeze(-1)
                    A0  = None
                    b0  = None
                elif (self.num_ineq == 0) and (self.num_eq != 0):
                    G0  = None
                    c0  = None
                    A0  = A[i]
                    b0  = b[i].squeeze(-1)
                else:
                    G0  = None
                    c0  = None
                    A0  = None
                    b0  = None

                # ---- lb_i, ub_i in 1D for this sample ----
                lb_i = lb[i]
                ub_i = ub[i]
                if lb_i.ndim == 2 and lb_i.shape[1] == 1:
                    lb_i = lb_i.squeeze(-1)
                if ub_i.ndim == 2 and ub_i.shape[1] == 1:
                    ub_i = ub_i.squeeze(-1)

                # ---- construct the Ipopt problem with box + (optional) G, A, c, b ----
                nlp = ConstrainedQP(
                    Q[i],
                    p[i].squeeze(-1),
                    lb=lb_i,
                    ub=ub_i,
                    G=G0,
                    c=c0,
                    A=A0,
                    b=b0,
                    tol=tol
                )

                nlp.add_option('tol', tol)
                nlp.add_option('print_level', 5)

                if init_mu is not None:
                    nlp.add_option('warm_start_init_point', 'yes')
                    nlp.add_option('warm_start_bound_push', 1e-20)
                    nlp.add_option('warm_start_bound_frac', 1e-20)
                    nlp.add_option('warm_start_slack_bound_push', 1e-20)
                    nlp.add_option('warm_start_slack_bound_frac', 1e-20)
                    nlp.add_option('warm_start_mult_bound_push', 1e-20)
                    nlp.add_option('mu_strategy', 'monotone')
                    nlp.add_option('mu_init', init_mu[i].squeeze().cpu().item())

                # ---- warm-start multipliers if provided ----
                if init_g is not None:
                    g = [x.item() for x in init_g[i].cpu()]
                else:
                    g = []

                if init_zl is not None:
                    zl = [x.item() for x in init_zl[i].cpu()]
                else:
                    zl = []

                if init_zu is not None:
                    zu = [x.item() for x in init_zu[i].cpu()]
                else:
                    zu = []

                start_time = time.time()
                y, info = nlp.solve(y0, lagrange=g, zl=zl, zu=zu)
                end_time = time.time()

                Y.append(y)
                iters.append(nlp.iters)
                total_time += (end_time - start_time)

            sols = np.array(Y)
            parallel_time = total_time / Q.shape[0]

        else:
            raise NotImplementedError
        
        
        
        return sols, total_time, parallel_time, np.array(iters).mean()
    
    # def merit_M(self,
    #         x, x1, x2, s,       # add s1 and s2 if there are bounds for inequality
    #         y, v, z1, z2, w,    # add w1 and w2 instead of w if there are bounds for inequality
    #         x1E, x2E, sE, yE, vE, z1E, z2E, wE, # add w1E and w2E instead of wE if there are bounds for inequality
    #         muP, muB, bad_x1, bad_x2, bad_z1, bad_z2):  

    #     B, n, device = x.shape[0], x.shape[1], x.device
    #     x = x.unsqueeze(-1) if x.dim()==2 else x        # [B,n,1]
    #     f_val = self.obj_fn(x)           
    #     n = self.num_var               # [B,1,1]
    #     lb_vals = self.lb.expand(B, n)    # [B,n]
    #     ub_vals = self.ub.expand(B, n)    # [B,n]
    #     muB = muB.unsqueeze(-1)
    #     if self.num_lb != 0:
    #         x1 = x1.unsqueeze(-1) if x1.dim()==2 else x1        # [B,n,1]
    #         x1E = x1E.unsqueeze(-1) if x1E.dim()==2 else x1E        # [B,n,1]
    #         z1 = z1.unsqueeze(-1) if z1.dim()==2 else z1        # [B,n,1]
    #         z1E = z1E.unsqueeze(-1) if z1E.dim()==2 else z1E        # [B,n,1]
    #         mask_lb = torch.isfinite(lb_vals).to(device) 
    #         mask_lb = mask_lb.unsqueeze(-1)

    #         mask_bad_x1  = bad_x1.unsqueeze(-1) if bad_x1.dim()==2 else bad_x1
    #         mask_bad_z1 = bad_z1.unsqueeze(-1) if bad_z1.dim()==2 else bad_z1
    #         mask_good_x1 = mask_lb & (~mask_bad_x1)
    #         mask_good_z1 = mask_lb & (~mask_bad_z1)

    #         eps_log = 1e-12      # something safely > 0

    #         safe_x1 = torch.where(mask_good_x1, x1 + muB, torch.ones_like(x1))
    #         safe_z1 = torch.where(mask_good_z1, z1 + muB, torch.ones_like(z1))

    #         t1 = ((z1E + x1E + muB) * torch.log(torch.clamp(safe_x1, min=eps_log))) \
    #             * mask_good_x1
    #         t2 = ((z1E + x1E + muB) * torch.log(torch.clamp(safe_z1, min=eps_log))) \
    #             * mask_good_z1

    #         # t1 = ((z1E + x1 + muB) * torch.log(x1 + muB)) * mask_lb
    #         # t2 = ((z1E + x1 + muB) * torch.log(z1 + muB)) * mask_lb
    #         t3 = ( z1 * (x1 + muB)             ) * mask_lb
    #         t4 = (          x1                 ) * mask_lb

    #         lb1 = -2 * muB * t1.sum(dim=1, keepdim=True)   # [B,1,1]
    #         lb2 = -1 * muB * t2.sum(dim=1, keepdim=True)
    #         lb3 =      t3.sum(dim=1, keepdim=True)
    #         lb4 =  2 * muB * t4.sum(dim=1, keepdim=True)

    #         lb_term_good = lb1 + lb2 + lb3 + lb4

    #         muA = muB*0.8               

    #         AL = (
    #             - z1E * x1
    #             + (x1**2) / (2*muA)
    #             + ((x1 + muA*(z1 - z1E))**2) / (2*muA)
    #         )                                              # [B,n,1]

    #         lb_term_fix = (AL * mask_bad_x1).sum(dim=1, keepdim=True)

    #         lb_term = lb_term_good + lb_term_fix

    #     if self.num_ub != 0:
    #         x2 = x2.unsqueeze(-1) if x2.dim()==2 else x2        # [B,n,1]
    #         x2E = x2E.unsqueeze(-1) if x2E.dim()==2 else x2E        # [B,n,1]
    #         z2 = z2.unsqueeze(-1) if z2.dim()==2 else z2        # [B,n,1]
    #         z2E = z2E.unsqueeze(-1) if z2E.dim()==2 else z2E        # [B,n,1]
    #         mask_ub = torch.isfinite(ub_vals).to(device)  
    #         mask_ub = mask_ub.unsqueeze(-1)

    #         mask_bad_x2  = bad_x2.unsqueeze(-1) if bad_x2.dim()==2 else bad_x2
    #         mask_bad_z2 = bad_z2.unsqueeze(-1) if bad_z2.dim()==2 else bad_z2
    #         mask_good_x2 = mask_ub & (~mask_bad_x2)
    #         mask_good_z2 = mask_ub & (~mask_bad_z2)

    #         eps_log = 1e-12      # something safely > 0

    #         safe_x2 = torch.where(mask_good_x2, x2 + muB, torch.ones_like(x2))
    #         safe_z2 = torch.where(mask_good_z2, z2 + muB, torch.ones_like(z2))

    #         t1 = ((z2E + x2E + muB) * torch.log(torch.clamp(safe_x2, min=eps_log))) \
    #             * mask_good_x2
    #         t2 = ((z2E + x2E + muB) * torch.log(torch.clamp(safe_z2, min=eps_log))) \
    #             * mask_good_z2

    #         # t1 = ((z2E + x2E + muB) * torch.log(x2 + muB)) * mask_ub
    #         # t2 = ((z2E + x2E + muB) * torch.log(z2 + muB)) * mask_ub
    #         t3 = ( z2 * (x2 + muB)             ) * mask_ub
    #         t4 = (             x2              ) * mask_ub

    #         ub1 = -2 * muB * t1.sum(dim=1, keepdim=True)   # [B,1,1]
    #         ub2 = -1 * muB * t2.sum(dim=1, keepdim=True)
    #         ub3 =      t3.sum(dim=1, keepdim=True)
    #         ub4 =  2 * muB * t4.sum(dim=1, keepdim=True)

    #         ub_term_good = ub1 + ub2 + ub3 + ub4

    #         muA      = muB*0.8                 # [B,1,1]
    #         AL2 = (
    #             - z2E * x2
    #             + (x2**2)    / (2 * muA)
    #             + ((x2 + muA * (z2 - z2E))**2) / (2 * muA)
    #         )                                               # [B,n,1]

    #         ub_term_fix = (AL2 * mask_bad_x2).sum(dim=1, keepdim=True)

    #         # 4) combine into final ub_term
    #         ub_term = ub_term_good + ub_term_fix

    #     if self.num_ineq != 0:
    #         s = s.unsqueeze(-1) if s.dim()==2 else s        # [B,n,1]
    #         sE = sE.unsqueeze(-1) if sE.dim()==2 else sE        # [B,n,1]
    #         w = w.unsqueeze(-1) if w.dim()==2 else w        # [B,n,1]
    #         wE = wE.unsqueeze(-1) if wE.dim()==2 else wE        # [B,n,1]
    #         y = y.unsqueeze(-1) if y.dim()==2 else y        # [B,n,1]
    #         yE = yE.unsqueeze(-1) if yE.dim()==2 else yE        # [B,n,1]
    #     if self.num_eq != 0:
    #         v = v.unsqueeze(-1) if v.dim()==2 else v       # [B,n,1]
    #         vE = vE.unsqueeze(-1) if vE.dim()==2 else vE       # [B,n,1]

    #     M_val = f_val + lb_term + ub_term

    #     return M_val

    def merit_M(self,
            x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA, bad_x1, bad_x2):  

        B, n, device = x.shape[0], x.shape[1], x.device
        #print(n)
        x = x.unsqueeze(-1) if x.dim()==2 else x        # [B,n,1]
        f_val = self.obj_fn(x)           
        n = self.num_var               # [B,1,1]
        lb_vals = self.lb.expand(B, n)    # [B,n]
        ub_vals = self.ub.expand(B, n)    # [B,n]
        muB = muB.unsqueeze(-1)

        #print(self.Q)

        ######## define x1 and x2 wrt to x inside this, I think the model is getting confused. 
        if self.num_lb != 0:
            mask_lb = torch.isfinite(lb_vals).to(device) 
            mask_lb = mask_lb.unsqueeze(-1)
            raw_x1 = (x - lb_vals.unsqueeze(-1))            # [B,n,1]
            x1     = torch.where(mask_lb, raw_x1,
                                torch.zeros_like(raw_x1))
            x1 = x1.unsqueeze(-1) if x1.dim()==2 else x1        # [B,n,1]
            x1E = x1E.unsqueeze(-1) if x1E.dim()==2 else x1E        # [B,n,1]
            z1 = z1.unsqueeze(-1) if z1.dim()==2 else z1        # [B,n,1]
            z1E = z1E.unsqueeze(-1) if z1E.dim()==2 else z1E        # [B,n,1]

            mask_bad_x1  = bad_x1.unsqueeze(-1) if bad_x1.dim()==2 else bad_x1
            #mask_bad_z1 = bad_z1.unsqueeze(-1) if bad_z1.dim()==2 else bad_z1
            mask_good_x1 = mask_lb & (~mask_bad_x1)
            mask_good_z1 = mask_lb #& (~mask_bad_z1)

            eps_log = 1e-20      # something safely > 0

            #print(z1.shape)

            safe_x1 = torch.where(mask_good_x1, (x1 + muB).clamp(eps_log), torch.ones_like(x1))
            safe_z1 = torch.where(mask_good_z1, (z1 + muB).clamp(eps_log), torch.ones_like(z1))

            t1 = ((z1E + x1E + muB) * torch.log(safe_x1)) \
                * mask_good_x1
            t2 = ((z1E + x1E + muB) * torch.log(safe_z1)) \
                * mask_good_x1

            t3 = ( z1 * (x1 + muB)             ) * mask_good_x1
            t4 = (          x1                 ) * mask_good_x1

            # print(t1)
            # print(t2)

            lb1 = -2 * muB * t1.sum(dim=1, keepdim=True)   # [B,1,1]
            lb2 = -1 * muB * t2.sum(dim=1, keepdim=True)
            lb3 =      t3.sum(dim=1, keepdim=True)
            lb4 =  2 * muB * t4.sum(dim=1, keepdim=True)

            lb_term_good = lb1 + lb2 + lb3 + lb4

            muA = muB*0.8               

            AL1 = (
                - z1E * x1
                + (x1**2) / (2*muA)
                + ((x1 + muA*(z1 - z1E))**2) / (2*muA)
            )                                              # [B,n,1]

            AL1 = torch.where(mask_bad_x1, AL1, torch.zeros_like(AL1))
            lb_term_fix = AL1.sum(dim=1, keepdim=True)

            lb_term = lb_term_good + lb_term_fix

        if self.num_ub != 0:
            mask_ub = torch.isfinite(ub_vals).to(device)  
            mask_ub = mask_ub.unsqueeze(-1)
            raw_x2 = (ub_vals.unsqueeze(-1) - x)         # [B,n,1]
            x2     = torch.where(mask_ub, raw_x2,
                                torch.zeros_like(raw_x2))
            x2 = x2.unsqueeze(-1) if x2.dim()==2 else x2        # [B,n,1]
            x2E = x2E.unsqueeze(-1) if x2E.dim()==2 else x2E        # [B,n,1]
            z2 = z2.unsqueeze(-1) if z2.dim()==2 else z2        # [B,n,1]
            z2E = z2E.unsqueeze(-1) if z2E.dim()==2 else z2E        # [B,n,1]

            mask_bad_x2  = bad_x2.unsqueeze(-1) if bad_x2.dim()==2 else bad_x2
            #mask_bad_z2 = bad_z2.unsqueeze(-1) if bad_z2.dim()==2 else bad_z2
            mask_good_x2 = mask_ub & (~mask_bad_x2)
            mask_good_z2 = mask_ub #& (~mask_bad_z2)

            eps_log = 1e-20      # something safely > 0

            safe_x2 = torch.where(mask_good_x2, (x2 + muB).clamp(eps_log), torch.ones_like(x2))
            safe_z2 = torch.where(mask_good_z2, (z2 + muB).clamp(eps_log), torch.ones_like(z2))

            t1 = ((z2E + x2E + muB) * torch.log(safe_x2)) \
                * mask_good_x2
            t2 = ((z2E + x2E + muB) * torch.log(safe_z2)) \
                * mask_good_x2

            # t1 = ((z2E + x2E + muB) * torch.log(x2 + muB)) * mask_ub
            # t2 = ((z2E + x2E + muB) * torch.log(z2 + muB)) * mask_ub
            t3 = ( z2 * (x2 + muB)             ) * mask_good_x2
            t4 = (             x2              ) * mask_good_x2

            ub1 = -2 * muB * t1.sum(dim=1, keepdim=True)   # [B,1,1]
            ub2 = -1 * muB * t2.sum(dim=1, keepdim=True)
            ub3 =      t3.sum(dim=1, keepdim=True)
            ub4 =  2 * muB * t4.sum(dim=1, keepdim=True)

            ub_term_good = ub1 + ub2 + ub3 + ub4

            muA      = muB*0.8                 # [B,1,1]
            AL2 = (
                - z2E * x2
                + (x2**2)    / (2 * muA)
                + ((x2 + muA * (z2 - z2E))**2) / (2 * muA)
            )                                               # [B,n,1]

            AL2 = torch.where(mask_bad_x2, AL2, torch.zeros_like(AL2))
            ub_term_fix = AL2.sum(dim=1, keepdim=True)

            # 4) combine into final ub_term
            ub_term = ub_term_good + ub_term_fix

        ineq_term = 0
        if self.num_ineq != 0:
            muP = muP.unsqueeze(-1) if muP.dim() == 2 else muP

            # Residual r_p = c(x) - s = (Gx - c) - s
            rp = self.ineq_resid(x) - (s.unsqueeze(-1) if s.dim() == 2 else s)     # [B,m_ineq,1]

            # - (c(x)-s)^T yE
            yE_ = yE.unsqueeze(-1) if yE.dim() == 2 else yE
            rp_t   = rp.transpose(1, 2).contiguous()                             # [B,1,m_ineq]
            yE_bmm = yE_.clone()                                                 # [B,m_ineq,1]

            # - (c(x)-s)^T yE
            t_p1 = -torch.bmm(rp_t, yE_bmm) 

            # (1/(2 μ^P)) ||c(x)-s||^2
            t_p2 = 0.5 / muP * torch.bmm(rp.transpose(1, 2), rp)                    # [B,1,1]

            # (1/(2 μ^P)) ||c(x)-s + μ^P (y - yE)||^2
            y_  = y.unsqueeze(-1) if y.dim()  == 2 else y
            shift_p = rp + muP * (y_ - yE_)                                         # [B,m_ineq,1]
            t_p3 = 0.5 / muP * torch.bmm(shift_p.transpose(1, 2), shift_p)          # [B,1,1]

            ineq_penalty = t_p1 + t_p2 + t_p3

            # ---- barrier for lower bound on s: (s1, w1) only ----
            # s1 = max(ε, s - s_lb) where s_lb defaults to 0 if not provided
            s_lb = getattr(self, 's_lb', torch.zeros(B, self.num_ineq, 1, device=device))
            mask_s_lb = torch.isfinite(s_lb)

            eps_log = 1e-20
            s_  = s if s.dim() == 3 else s.unsqueeze(-1)                             # [B,m_ineq,1]
            s1  = torch.where(mask_s_lb, (s_ - s_lb).clamp(min=eps_log), torch.zeros_like(s_))
            w1_ = w1 if (w1 is None or w1.dim() == 3) else w1.unsqueeze(-1)
            w1_ = torch.zeros_like(s1) if w1 is None else w1_

            s1E_ = s1E if (s1E is None or s1E.dim() == 3) else s1E.unsqueeze(-1)
            s1E_ = torch.zeros_like(s1) if s1E is None else s1E_
            w1E_ = w1E if (w1E is None or w1E.dim() == 3) else w1E.unsqueeze(-1)
            w1E_ = torch.zeros_like(s1) if w1E is None else w1E_

            safe_s1 = torch.where(mask_s_lb, (s1 + muB).clamp(eps_log), torch.ones_like(s1))
            safe_w1 = torch.where(mask_s_lb, (w1_ + muB).clamp(eps_log), torch.ones_like(w1_))

            u1 = ((w1E_ + s1E_ + muB) * torch.log(safe_s1)) * mask_s_lb
            u2 = ((w1E_ + s1E_ + muB) * torch.log(safe_w1)) * mask_s_lb
            u3 = (w1_ * (s1 + muB)) * mask_s_lb
            u4 = s1 * mask_s_lb

            ineq_barrier = (-2 * muB) * u1.sum(dim=1, keepdim=True) \
                        + (-1 * muB) * u2.sum(dim=1, keepdim=True) \
                        + u3.sum(dim=1, keepdim=True) \
                        + (2 * muB) * u4.sum(dim=1, keepdim=True)

            ineq_term = ineq_penalty + ineq_barrier

        eq_term = 0
        if self.num_eq != 0:
            muP = muP.unsqueeze(-1) if muP.dim() == 2 else muP
            re = self.eq_resid(x)                              # [B,m_eq,1] = Ax - b

            vE_ = vE.unsqueeze(-1) if vE.dim() == 2 else vE
            v_  = v.unsqueeze(-1)  if v.dim()  == 2 else v

            e1 = -torch.bmm(re.transpose(1, 2), vE_)           # - (Ax-b)^T vE
            e2 = (0.5 * torch.bmm(re.transpose(1, 2), re)) / muP # (1/(2 μP)) ||Ax - b||^2

            shift_e = re + muP * (v_ - vE_)                    # Ax - b + μP (v - vE)
            e3 = (0.5 * torch.bmm(shift_e.transpose(1, 2), shift_e)) / muP

            eq_term = e1 + e2 + e3

        M_val = f_val + lb_term + ub_term + ineq_term + eq_term

        return M_val
    
    def merit_M_indi(self,
            x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA, bad_x1, bad_x2):  

        B, n, device = x.shape[0], x.shape[1], x.device
        #print(n)
        x = x.unsqueeze(-1) if x.dim()==2 else x        # [B,n,1]
        f_val = self.obj_fn(x)           
        n = self.num_var               # [B,1,1]
        lb_vals = self.lb.expand(B, n)    # [B,n]
        ub_vals = self.ub.expand(B, n)    # [B,n]
        muB = muB.unsqueeze(-1)

        #print(self.Q)

        ######## define x1 and x2 wrt to x inside this, I think the model is getting confused. 
        if self.num_lb != 0:
            mask_lb = torch.isfinite(lb_vals).to(device) 
            mask_lb = mask_lb.unsqueeze(-1)
            raw_x1 = (x - lb_vals.unsqueeze(-1))            # [B,n,1]
            x1     = torch.where(mask_lb, raw_x1,
                                torch.zeros_like(raw_x1))
            x1 = x1.unsqueeze(-1) if x1.dim()==2 else x1        # [B,n,1]
            x1E = x1E.unsqueeze(-1) if x1E.dim()==2 else x1E        # [B,n,1]
            z1 = z1.unsqueeze(-1) if z1.dim()==2 else z1        # [B,n,1]
            z1E = z1E.unsqueeze(-1) if z1E.dim()==2 else z1E        # [B,n,1]

            mask_bad_x1  = bad_x1.unsqueeze(-1) if bad_x1.dim()==2 else bad_x1
            #mask_bad_z1 = bad_z1.unsqueeze(-1) if bad_z1.dim()==2 else bad_z1
            mask_good_x1 = mask_lb & (~mask_bad_x1)
            mask_good_z1 = mask_lb #& (~mask_bad_z1)

            eps_log = 1e-20      # something safely > 0

            #print(z1.shape)

            safe_x1 = torch.where(mask_good_x1, (x1 + muB).clamp(eps_log), torch.ones_like(x1))
            safe_z1 = torch.where(mask_good_z1, (z1 + muB).clamp(eps_log), torch.ones_like(z1))

            t1 = ((z1E + x1E + muB) * torch.log(safe_x1)) \
                * mask_good_x1
            t2 = ((z1E + x1E + muB) * torch.log(safe_z1)) \
                * mask_good_x1

            t3 = ( z1 * (x1 + muB)             ) * mask_good_x1
            t4 = (          x1                 ) * mask_good_x1

            # print(t1)
            # print(t2)

            lb1 = -2 * muB * t1.sum(dim=1, keepdim=True)   # [B,1,1]
            lb2 = -1 * muB * t2.sum(dim=1, keepdim=True)
            lb3 =      t3.sum(dim=1, keepdim=True)
            lb4 =  2 * muB * t4.sum(dim=1, keepdim=True)

            lb_term_good = lb1 + lb2 + lb3 + lb4

            muA = muB*0.8               

            AL1 = (
                - z1E * x1
                + (x1**2) / (2*muA)
                + ((x1 + muA*(z1 - z1E))**2) / (2*muA)
            )                                              # [B,n,1]

            AL1 = torch.where(mask_bad_x1, AL1, torch.zeros_like(AL1))
            lb_term_fix = AL1.sum(dim=1, keepdim=True)

            lb_term = lb_term_good + lb_term_fix

        if self.num_ub != 0:
            mask_ub = torch.isfinite(ub_vals).to(device)  
            mask_ub = mask_ub.unsqueeze(-1)
            raw_x2 = (ub_vals.unsqueeze(-1) - x)         # [B,n,1]
            x2     = torch.where(mask_ub, raw_x2,
                                torch.zeros_like(raw_x2))
            x2 = x2.unsqueeze(-1) if x2.dim()==2 else x2        # [B,n,1]
            x2E = x2E.unsqueeze(-1) if x2E.dim()==2 else x2E        # [B,n,1]
            z2 = z2.unsqueeze(-1) if z2.dim()==2 else z2        # [B,n,1]
            z2E = z2E.unsqueeze(-1) if z2E.dim()==2 else z2E        # [B,n,1]

            mask_bad_x2  = bad_x2.unsqueeze(-1) if bad_x2.dim()==2 else bad_x2
            #mask_bad_z2 = bad_z2.unsqueeze(-1) if bad_z2.dim()==2 else bad_z2
            mask_good_x2 = mask_ub & (~mask_bad_x2)
            mask_good_z2 = mask_ub #& (~mask_bad_z2)

            eps_log = 1e-20      # something safely > 0

            safe_x2 = torch.where(mask_good_x2, (x2 + muB).clamp(eps_log), torch.ones_like(x2))
            safe_z2 = torch.where(mask_good_z2, (z2 + muB).clamp(eps_log), torch.ones_like(z2))

            t1 = ((z2E + x2E + muB) * torch.log(safe_x2)) \
                * mask_good_x2
            t2 = ((z2E + x2E + muB) * torch.log(safe_z2)) \
                * mask_good_x2

            # t1 = ((z2E + x2E + muB) * torch.log(x2 + muB)) * mask_ub
            # t2 = ((z2E + x2E + muB) * torch.log(z2 + muB)) * mask_ub
            t3 = ( z2 * (x2 + muB)             ) * mask_good_x2
            t4 = (             x2              ) * mask_good_x2

            ub1 = -2 * muB * t1.sum(dim=1, keepdim=True)   # [B,1,1]
            ub2 = -1 * muB * t2.sum(dim=1, keepdim=True)
            ub3 =      t3.sum(dim=1, keepdim=True)
            ub4 =  2 * muB * t4.sum(dim=1, keepdim=True)

            ub_term_good = ub1 + ub2 + ub3 + ub4

            muA      = muB*0.8                 # [B,1,1]
            AL2 = (
                - z2E * x2
                + (x2**2)    / (2 * muA)
                + ((x2 + muA * (z2 - z2E))**2) / (2 * muA)
            )                                               # [B,n,1]

            AL2 = torch.where(mask_bad_x2, AL2, torch.zeros_like(AL2))
            ub_term_fix = AL2.sum(dim=1, keepdim=True)

            # 4) combine into final ub_term
            ub_term = ub_term_good + ub_term_fix

        ineq_term = 0
        if self.num_ineq != 0:
            muP = muP.unsqueeze(-1) if muP.dim() == 2 else muP

            # Residual r_p = c(x) - s = (Gx - c) - s
            rp = self.ineq_resid(x) - (s.unsqueeze(-1) if s.dim() == 2 else s)     # [B,m_ineq,1]

            # - (c(x)-s)^T yE
            yE_ = yE.unsqueeze(-1) if yE.dim() == 2 else yE
            rp_t   = rp.transpose(1, 2).contiguous()                             # [B,1,m_ineq]
            yE_bmm = yE_.clone()                                                 # [B,m_ineq,1]

            # - (c(x)-s)^T yE
            t_p1 = -torch.bmm(rp_t, yE_bmm) 

            # (1/(2 μ^P)) ||c(x)-s||^2
            t_p2 = 0.5 / muP * torch.bmm(rp.transpose(1, 2), rp)                    # [B,1,1]

            # (1/(2 μ^P)) ||c(x)-s + μ^P (y - yE)||^2
            y_  = y.unsqueeze(-1) if y.dim()  == 2 else y
            shift_p = rp + muP * (y_ - yE_)                                         # [B,m_ineq,1]
            t_p3 = 0.5 / muP * torch.bmm(shift_p.transpose(1, 2), shift_p)          # [B,1,1]

            ineq_penalty = t_p1 + t_p2 + t_p3

            # ---- barrier for lower bound on s: (s1, w1) only ----
            # s1 = max(ε, s - s_lb) where s_lb defaults to 0 if not provided
            s_lb = getattr(self, 's_lb', torch.zeros(B, self.num_ineq, 1, device=device))
            mask_s_lb = torch.isfinite(s_lb)

            eps_log = 1e-20
            s_  = s if s.dim() == 3 else s.unsqueeze(-1)                             # [B,m_ineq,1]
            s1  = torch.where(mask_s_lb, (s_ - s_lb).clamp(min=eps_log), torch.zeros_like(s_))
            w1_ = w1 if (w1 is None or w1.dim() == 3) else w1.unsqueeze(-1)
            w1_ = torch.zeros_like(s1) if w1 is None else w1_

            s1E_ = s1E if (s1E is None or s1E.dim() == 3) else s1E.unsqueeze(-1)
            s1E_ = torch.zeros_like(s1) if s1E is None else s1E_
            w1E_ = w1E if (w1E is None or w1E.dim() == 3) else w1E.unsqueeze(-1)
            w1E_ = torch.zeros_like(s1) if w1E is None else w1E_

            safe_s1 = torch.where(mask_s_lb, (s1 + muB).clamp(eps_log), torch.ones_like(s1))
            safe_w1 = torch.where(mask_s_lb, (w1_ + muB).clamp(eps_log), torch.ones_like(w1_))

            u1 = ((w1E_ + s1E_ + muB) * torch.log(safe_s1)) * mask_s_lb
            u2 = ((w1E_ + s1E_ + muB) * torch.log(safe_w1)) * mask_s_lb
            u3 = (w1_ * (s1 + muB)) * mask_s_lb
            u4 = s1 * mask_s_lb

            ineq_barrier = (-2 * muB) * u1.sum(dim=1, keepdim=True) \
                        + (-1 * muB) * u2.sum(dim=1, keepdim=True) \
                        + u3.sum(dim=1, keepdim=True) \
                        + (2 * muB) * u4.sum(dim=1, keepdim=True)

            ineq_term = ineq_penalty + ineq_barrier

        eq_term = 0
        if self.num_eq != 0:
            muP = muP.unsqueeze(-1) if muP.dim() == 2 else muP
            re = self.eq_resid(x)                              # [B,m_eq,1] = Ax - b

            vE_ = vE.unsqueeze(-1) if vE.dim() == 2 else vE
            v_  = v.unsqueeze(-1)  if v.dim()  == 2 else v

            e1 = -torch.bmm(re.transpose(1, 2), vE_)           # - (Ax-b)^T vE
            e2 = (0.5 * torch.bmm(re.transpose(1, 2), re)) / muP # (1/(2 μP)) ||Ax - b||^2

            shift_e = re + muP * (v_ - vE_)                    # Ax - b + μP (v - vE)
            e3 = (0.5 * torch.bmm(shift_e.transpose(1, 2), shift_e)) / muP

            eq_term = e1 + e2 + e3

        M_val = f_val + lb_term + ub_term + ineq_term + eq_term

        return f_val, lb1, lb2, lb3, lb4, lb_term_fix, ub1, ub2, ub3, ub4, ub_term_fix, e1, e2, e3
    
    # def merit_grad_M(self,
    #         x, s,      
    #         y, v, z1, z2, w1, w2,    
    #         x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
    #         muP, muB, muA, bad_x1, bad_x2, bad_z1, bad_z2):  

    #     B, device = x.shape[0], x.device
    #     n = self.num_var
    #     x = x.unsqueeze(-1) if x.dim()==2 else x        # [B,n,1]
    #     grad_x = self.obj_grad(x)     
    #     lb_vals = self.lb.expand(B, n)    # [B,n]
    #     ub_vals = self.ub.expand(B, n)    # [B,n] 
    #     muB = muB.unsqueeze(-1)                     # [B,1,1]
    #     muA = 0.8*muB
    #     eps_div = 1e-18
    #     if self.num_lb != 0:
    #         mask_lb = torch.isfinite(lb_vals).to(device)         # [1,n,1] boolean
    #         mask_lb = mask_lb.unsqueeze(-1)                # [B,n,1]
    #         raw_x1 = (x - lb_vals)            # [B,n,1]
    #         x1     = torch.where(mask_lb, raw_x1,
    #                             torch.zeros_like(raw_x1))
    #         x1 = x1.unsqueeze(-1) if x1.dim()==2 else x1        # [B,n,1]
    #         x1E = x1E.unsqueeze(-1) if x1E.dim()==2 else x1E        # [B,n,1]
    #         z1 = z1.unsqueeze(-1) if z1.dim()==2 else z1        # [B,n,1]
    #         z1E = z1E.unsqueeze(-1) if z1E.dim()==2 else z1E        # [B,n,1]
    #         e_x1 = torch.ones_like(x1)              # [B,m,1]
    #         lb_vals = self.lb.expand(B, n)         # [1,n,1]

    #         mask_bad_x1  = bad_x1.unsqueeze(-1) if bad_x1.dim()==2 else bad_x1
    #         mask_bad_z1 = bad_z1.unsqueeze(-1) if bad_z1.dim()==2 else bad_z1
    #         mask_good_x1 = mask_lb & (~mask_bad_x1)
    #         mask_good_z1 = mask_lb & (~mask_bad_z1)

    #         e_x1    = torch.ones_like(x1)                        # [B,n,1]
    #         inv_X1mu = 1.0/(x1 + muB).clamp_min(eps_div)                            # [B,n,1]
    #         inv_Z1mu = 1.0/(z1 + muB).clamp_min(eps_div)                            # [B,n,1]

    #         # raw_grad_z1_linear= x1 + muB*e_x1

    #         # # raw gradients for the lower-slack block
    #         # raw_grad_z1_barrier = (
    #         # - muB * inv_Z1mu * (z1E + x1E + muB*e_x1)
    #         # )  

    #         x1_mu = x1 + muB * e_x1                               # [B,n,1]
    #         z1_mu = z1 + muB * e_x1                               # [B,n,1]

    #         Dz1   = x1_mu * inv_Z1mu                              # = (x1+μB)/(z1+μB)

    #         # π1^Z = μB * (X1^μ)^{-1} * (z1^E - x1 + x1^E)
    #         pi1z  = muB * inv_X1mu * (z1E - x1 + x1E)            # [B,n,1]

    #         # --- gradient for the good-x1 (barrier) case --------------------------------
    #         # use Dz1 * (z1 - pi1z); this is the sign in the paper and gives descent with z ← z − α*grad
    #         grad_z1_barrier = Dz1 * (z1 - pi1z)
            
    #         raw_grad_x1_linear = z1 + 2*muB*e_x1                                               

    #         raw_grad_x1_barrier = (
    #         - 2*muB * inv_X1mu * (z1E + x1E + muB*e_x1)
    #         )      





    #         raw_grad_x1_quadratic = (
    #             - z1E
    #             + (2*x1 + muA * (z1 - z1E)) / muA
    #         )    

            
    #         raw_grad_z1_quadratic = muA + (x1 + muA * (z1 - z1E))/muA                                    

    #         # now zero out any component j for which ℓ_j = -∞
    #         #+ raw_grad_x1_barrier * mask_good_x1 + raw_grad_x1_quadratic * mask_bad_x1
    #         #+ raw_grad_z1_barrier * mask_good_z1 + raw_grad_z1_quadratic * mask_bad_x1

    #         #grad_x1 = (raw_grad_x1_linear + raw_grad_x1_barrier) * mask_good_x1 + raw_grad_x1_quadratic * mask_bad_x1
    #         grad_x1 = (-z1) * mask_good_x1 + raw_grad_x1_quadratic * mask_bad_x1 #+ raw_grad_x1_barrier * mask_lb
    #         #grad_z1 = (raw_grad_z1_linear + raw_grad_z1_barrier) * mask_good_x1 + raw_grad_z1_quadratic * mask_bad_x1 #+ raw_grad_z1_barrier * mask_lb
    #         grad_z1 = (grad_z1_barrier) * mask_good_x1 + raw_grad_z1_quadratic * mask_bad_x1 
    #     if self.num_ub != 0:
    #         mask_ub = torch.isfinite(ub_vals).to(device)     # [1,n,1]
    #         mask_ub = mask_ub.unsqueeze(-1)              # [B,n,1]
    #         raw_x2 = (ub_vals - x)         # [B,n,1]
    #         x2     = torch.where(mask_ub, raw_x2,
    #                             torch.zeros_like(raw_x2))
    #         x2 = x2.unsqueeze(-1) if x2.dim()==2 else x2        # [B,n,1]
    #         x2E = x2E.unsqueeze(-1) if x2E.dim()==2 else x2E        # [B,n,1]
    #         z2 = z2.unsqueeze(-1) if z2.dim()==2 else z2        # [B,n,1]
    #         z2E = z2E.unsqueeze(-1) if z2E.dim()==2 else z2E        # [B,n,1]
    #         ub_vals = self.ub.expand(B, n)   # [1,n,1]

    #         mask_bad_x2  = bad_x2.unsqueeze(-1) if bad_x2.dim()==2 else bad_x2
    #         mask_bad_z2 = bad_z2.unsqueeze(-1) if bad_z2.dim()==2 else bad_z2
    #         mask_good_x2 = mask_ub & (~mask_bad_x2)
    #         mask_good_z2 = mask_ub & (~mask_bad_z2)

    #         # 2) compute raw gradients exactly as before
    #         e_x2    = torch.ones_like(x2)                    # [B,n,1]
    #         inv_X2mu = 1.0/(x2 + muB).clamp_min(eps_div)                        # [B,n,1]
    #         inv_Z2mu = 1.0/(z2 + muB).clamp_min(eps_div)                        # [B,n,1]

    #         # raw_grad_z2_linear = x2 + muB*e_x2

    #         # # raw gradients for the lower-slack block
    #         # raw_grad_z2_barrier = (
    #         # - muB * inv_Z2mu * (z2E + x2E + muB*e_x2)
    #         # )  

    #         x2_mu = x2 + muB * e_x2                                     # [B,n,1]
    #         z2_mu = z2 + muB * e_x2                                     # [B,n,1]

    #         Dz2  = x2_mu * inv_Z2mu                                     # = (x2+μB)/(z2+μB)
    #         pi2z = muB * inv_X2mu * (z2E - x2 + x2E)                    # [B,n,1]

    #         # --- gradient used in the barrier (good-x2) branch --------------------------
    #         # Use Dz2 * (z2 - pi2z); with update z2 ← z2 − α*grad_z2 this moves z2 → pi2z.
    #         grad_z2_barrier = Dz2 * (z2 - pi2z)
 
            
    #         raw_grad_x2_linear = z2 + 2*muB*e_x2                                               

    #         raw_grad_x2_barrier = (
    #         - 2*muB * inv_X2mu * (z2E + x2E + muB*e_x2)
    #         )           

    #         raw_grad_x2_quadratic = (
    #             - z2E
    #             + (2*x2 + muA * (z2 - z2E)) / muA
    #         )   
            
    #         raw_grad_z2_quadratic = muA + (x2 + muA * (z2 - z2E))/muA      

    #         # 3) zero out any coordinate j with u_j = +∞ 
    #         #+ raw_grad_x2_barrier * mask_lb
    #         #+ raw_grad_z2_barrier * mask_lb

    #         #grad_x2 = (raw_grad_x2_linear + raw_grad_x2_barrier) * mask_good_x2 + raw_grad_x2_quadratic * mask_bad_x2
    #         grad_x2 = (-z2) * mask_good_x2 + raw_grad_x2_quadratic * mask_bad_x2
    #         #grad_z2 = (raw_grad_z2_linear + raw_grad_z2_barrier) * mask_good_x2 + raw_grad_z2_quadratic * mask_bad_x2
    #         grad_z2 = (grad_z2_barrier) * mask_good_x2 + raw_grad_z2_quadratic * mask_bad_x2

    #     if self.num_ineq != 0:
    #         s  = s.unsqueeze(-1)  if s.dim()  == 2 else s      # [B,m_ineq,1]
    #         y  = y.unsqueeze(-1)  if y.dim()  == 2 else y      # [B,m_ineq,1]
    #         yE = yE.unsqueeze(-1) if yE.dim() == 2 else yE     # [B,m_ineq,1]
    #         w1 = w1.unsqueeze(-1) if w1.dim() == 2 else w1     # [B,m_ineq,1]
    #         w1E = w1E.unsqueeze(-1) if w1E.dim() == 2 else w1E # [B,m_ineq,1]
    #         muP = muP.unsqueeze(-1) if muP.dim() == 2 else muP # [B,1,1]

    #         # residual r_p = c(x) - s with c(x) := Gx - c
    #         G = self.G
    #         c_vec = self.c
    #         rp = self.ineq_resid(x) - s                         # [B,m_ineq,1]

    #         # D_Y = μ^P I,  π^Y = y^E - (1/μ^P) * (c(x) - s)
    #         # => grad_y = D_Y (y - π^Y) = μ^P (y - yE) + (c(x) - s)
    #         grad_y = muP * (y - yE) + rp                        # [B,m_ineq,1]

    #         # grad_s = y - w1  (no w2 term as you requested)
    #         # If you only want w1 where s has a finite lower bound, gate with a mask:
    #         s_lb = getattr(self, 's_lb', torch.zeros(B, self.num_ineq, 1, device=device))
    #         mask_s_lb = torch.isfinite(s_lb)                    # [B,m_ineq,1]
    #         grad_s = y - (w1 * mask_s_lb)                       # [B,m_ineq,1]

    #         # ----- w1 block:  -D1^W (π1^W - w1)  => grad_w1 = D1^W (w1 - π1^W)
    #         # D1^W = S1^μ (W1^μ)^(-1),  S1^μ = diag(s1 + μB), W1^μ = diag(w1 + μB)
    #         # π1^W = μB (S1^μ)^(-1) (w1^E - s1 + s1^E)
    #         eps_div = 1e-18
    #         muB = muB.unsqueeze(-1) if muB.dim() == 2 else muB  # [B,1,1]

    #         # s1 = s - s_lb (only where lb is finite)
    #         s1 = torch.where(mask_s_lb, (s - s_lb), torch.zeros_like(s))
    #         s1E = s1E.unsqueeze(-1) if s1E.dim() == 2 else s1E  # [B,m_ineq,1]

    #         S1mu = (s1 + muB).clamp_min(eps_div)
    #         W1mu = (w1 + muB).clamp_min(eps_div)

    #         Dw1   = S1mu / W1mu                                  # elementwise
    #         pi1_w = muB * (w1E - s1 + s1E) / S1mu                # elementwise

    #         grad_w1 = (Dw1 * (w1 - pi1_w)) * mask_s_lb          # [B,m_ineq,1]

    #         # ----- add the missing coupling in grad_x from inequalities: -J^T y = -G^T y
    #         grad_x = grad_x - torch.bmm(G.transpose(1, 2), y)    # [B,n,1]
    #     if self.num_eq != 0:
    #         v  = v.unsqueeze(-1)  if v.dim()  == 2 else v        # [B,m_eq,1]
    #         vE = vE.unsqueeze(-1) if vE.dim() == 2 else vE       # [B,m_eq,1]
    #         # If you keep muA passed in, use that; otherwise your code sets muA = 0.8*muB above
    #         muA = muA.unsqueeze(-1) if muA.dim() == 2 else muA   # [B,1,1]

    #         A = self.A
    #         re = self.eq_resid(x)                                # [B,m_eq,1] = Ax - b

    #         # D_A = μ^A I,  π^V = v^E - (1/μ^A) (Ax - b)
    #         # => grad_v = D_A (v - π^V) = μ^A (v - vE) + (Ax - b)
    #         grad_v = muA * (v - vE) + re                         # [B,m_eq,1]

    #         # add the missing coupling in grad_x: -A^T v
    #         grad_x = grad_x - torch.bmm(A.transpose(1, 2), v)    # [B,n,1]
    #     grad_x = grad_x + grad_x1 - grad_x2


    #     return grad_x, grad_s, grad_y, grad_v, grad_z1, grad_z2, grad_w1
    
    def merit_grad_M(self,
            x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA, bad_x1, bad_x2):  

        B, device = x.shape[0], x.device
        n = self.num_var
        x = x.unsqueeze(-1) if x.dim()==2 else x        # [B,n,1]
        grad_x = self.obj_grad(x)                       # [B,n,1]

        lb_vals = self.lb.expand(B, n)    # [B,n]
        ub_vals = self.ub.expand(B, n)    # [B,n] 
        muB = muB.unsqueeze(-1) if muB.dim()==2 else muB   # [B,1,1]
        muA = muA.unsqueeze(-1) if muA.dim()==2 else muA   # [B,1,1]
        eps_div = 1e-18
        sigma = 0.8

        # ------------------------ lower-box (x1,z1) ------------------------
        if self.num_lb != 0:
            mask_lb = torch.isfinite(lb_vals).to(device).unsqueeze(-1)   # [B,n,1]
            raw_x1 = (x - lb_vals.unsqueeze(-1))                                       # [B,n,1]
            x1     = torch.where(mask_lb, raw_x1, torch.zeros_like(raw_x1))
            x1 = x1 if x1.dim()==3 else x1.unsqueeze(-1)
            x1E = x1E if x1E.dim()==3 else x1E.unsqueeze(-1)
            z1  = z1  if z1.dim()==3  else z1.unsqueeze(-1)
            z1E = z1E if z1E.dim()==3 else z1E.unsqueeze(-1)

            e_x1    = torch.ones_like(x1)
            inv_X1mu = 1.0/(x1 + muB).clamp_min(eps_div)
            inv_Z1mu = 1.0/(z1 + muB).clamp_min(eps_div)

            x1_mu = x1 + muB * e_x1
            z1_mu = z1 + muB * e_x1

            Dz1   = x1_mu * inv_Z1mu
            pi1z  = muB * inv_X1mu * (z1E - x1 + x1E)

            # good/bad masks (your logic)
            mask_bad_x1  = bad_x1 if bad_x1.dim()==3 else bad_x1.unsqueeze(-1)
            #mask_bad_z1  = bad_z1 if bad_z1.dim()==3 else bad_z1.unsqueeze(-1)
            mask_good_x1 = mask_lb & (~mask_bad_x1)
            mask_good_z1 = mask_lb #& (~mask_bad_z1)

            grad_z1_barrier = Dz1 * (z1 - pi1z)

            raw_grad_x1_quadratic = (
                - z1E
                + (2*x1 + muA * (z1 - z1E)) / muA
            )
            raw_grad_z1_quadratic = muA + (x1 + muA * (z1 - z1E))/muA

            grad_x1 = (-z1) * mask_good_x1 + raw_grad_x1_quadratic * mask_bad_x1
            grad_z1 = (grad_z1_barrier) * mask_good_x1 + raw_grad_z1_quadratic * mask_bad_x1

        # ------------------------ upper-box (x2,z2) ------------------------
        if self.num_ub != 0:
            mask_ub = torch.isfinite(ub_vals).to(device).unsqueeze(-1)   # [B,n,1]
            raw_x2 = (ub_vals.unsqueeze(-1) - x)
            x2     = torch.where(mask_ub, raw_x2, torch.zeros_like(raw_x2))
            x2 = x2 if x2.dim()==3 else x2.unsqueeze(-1)
            x2E = x2E if x2E.dim()==3 else x2E.unsqueeze(-1)
            z2  = z2  if z2.dim()==3  else z2.unsqueeze(-1)
            z2E = z2E if z2E.dim()==3 else z2E.unsqueeze(-1)

            e_x2    = torch.ones_like(x2)
            inv_X2mu = 1.0/(x2 + muB).clamp_min(eps_div)
            inv_Z2mu = 1.0/(z2 + muB).clamp_min(eps_div)

            x2_mu = x2 + muB * e_x2
            z2_mu = z2 + muB * e_x2

            Dz2  = x2_mu * inv_Z2mu
            pi2z = muB * inv_X2mu * (z2E - x2 + x2E)

            mask_bad_x2  = bad_x2 if bad_x2.dim()==3 else bad_x2.unsqueeze(-1)
            #mask_bad_z2  = bad_z2 if bad_z2.dim()==3 else bad_z2.unsqueeze(-1)
            mask_good_x2 = mask_ub & (~mask_bad_x2)
            mask_good_z2 = mask_ub #& (~mask_bad_z2)

            grad_z2_barrier = Dz2 * (z2 - pi2z)

            raw_grad_x2_quadratic = (
                - z2E
                + (2*x2 + muA * (z2 - z2E)) / muA
            )
            raw_grad_z2_quadratic = muA + (x2 + muA * (z2 - z2E))/muA

            grad_x2 = (-z2) * mask_good_x2 + raw_grad_x2_quadratic * mask_bad_x2
            grad_z2 = (grad_z2_barrier) * mask_good_x2 + raw_grad_z2_quadratic * mask_bad_x2

        # ------------------------ inequality (s, y, w1) -------------------
        grad_s = grad_y = grad_w1 = None
        if self.num_ineq != 0:
            s   = s   if s.dim()==3   else s.unsqueeze(-1)     # [B,m,1]
            y   = y   if y.dim()==3   else y.unsqueeze(-1)
            yE  = yE  if yE.dim()==3  else yE.unsqueeze(-1)
            w1  = w1  if w1.dim()==3  else w1.unsqueeze(-1)
            w1E = w1E if w1E.dim()==3 else w1E.unsqueeze(-1)
            muP = muP if muP.dim()==3 else muP.unsqueeze(-1)   # [B,1,1]

            # residual rp = c(x) - s, with c(x)=Gx - c
            rp = self.ineq_resid(x) - s                        # [B,m,1]

            # π^Y = y^E - (1/μ^P)(c(x)-s)
            # D_Y(y-π^Y) = μ^P(y-yE) + (c(x)-s)
            DY_term = muP * (y - yE) + rp

            # Build s1, masks (lower bound on s; default 0)
            s_lb = getattr(self, 's_lb', torch.zeros_like(s))
            mask_s_lb = torch.isfinite(s_lb)                   # [B,m,1]
            s1  = torch.where(mask_s_lb, (s - s_lb), torch.zeros_like(s))
            s1E = s1E if s1E.dim()==3 else s1E.unsqueeze(-1)

            # π^W (lower only): μ^B (S1^μ)^{-1}(w1^E - s1 + s1^E)
            S1mu = (s1 + muB).clamp_min(eps_div)
            W1mu = (w1 + muB).clamp_min(eps_div)
            pi1_w = muB * (w1E - s1 + s1E) / S1mu

            # D_w1 = S1^μ (W1^μ)^{-1};  D_W ≈ (D_w1)^{-1} on lower-bound coordinates
            # i.e., diag( (W1^μ)/(S1^μ) ) where the lower bound exists, else 0
            DW = torch.where(mask_s_lb, (S1mu / W1mu), torch.zeros_like(S1mu))

            # ---- modified blocks ----
            # grad_s :  y - π^W
            grad_s  = y - pi1_w

            # grad_y :  D_Y(y-π^Y) + D_W(y-π^W)
            grad_y  = DY_term + DW * (y - pi1_w)

            # grad_w1 (unchanged): D_w1 (w1 - π^W) on lb coords
            Dw1 = S1mu / W1mu
            grad_w1 = (Dw1 * (w1 - pi1_w)) * mask_s_lb

            # coupling in x from inequalities: -G^T y
            Gt = (self.G.transpose(1, 2) if self.G.dim()==3
                  else self.G.t().unsqueeze(0).expand(B, -1, -1))
            grad_x = grad_x - torch.bmm(Gt, y)

        # ------------------------ equality (v) + x-coupling ----------------
        grad_v = None
        if self.num_eq != 0:
            muP = muP if muP.dim()==3 else muP.unsqueeze(-1)   # [B,1,1]
            v  = v  if v.dim()==3  else v.unsqueeze(-1)        # [B,meq,1]
            vE = vE if vE.dim()==3 else vE.unsqueeze(-1)
            re = self.eq_resid(x)                              # [B,meq,1] = Ax-b

            # DV_term = muP * (v - vE) + re

            # grad_v = D_A (v - π^V) = μ^A(v - vE) + (Ax - b)
            grad_v = muP * (v - vE) + re



            # Use π^V in grad_x coupling: -A^T π^V  (NOT -A^T v)
            piV = vE - re / muA
            At = (self.A.transpose(1, 2) if self.A.dim()==3
                  else self.A.t().unsqueeze(0).expand(B, -1, -1))
            grad_x = grad_x - torch.bmm(At, v)
        
        grad_x = grad_x - pi1z + pi2z

        # add box contributions to grad_x
        # if self.num_lb != 0:
        #     grad_x = grad_x + grad_x1
        # if self.num_ub != 0:
        #     grad_x = grad_x - grad_x2

        return grad_x, grad_s, grad_y, grad_v, grad_z1, grad_z2, grad_w1

    
    def merit_hess_inv_M(self,
            x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB,  muA, bad_x1, bad_x2, bad_z1, bad_z2                 # muB used
    ):
        """
        Batched inverse Hessian  H⁻¹  of the 3×3 block
            [  Q        I      -I ]
        H = [  I      D₁(μ)     0 ]
            [ -I        0    D₂(μ) ]

        where
            Q   = problem.Q  (B,n,n)
            D₁ᵢᵢ = 2 μᴮ / (z₁ᵢ + μᴮ)²
            D₂ᵢᵢ = 2 μᴮ / (z₂ᵢ + μᴮ)² .

        Returns
            H_inv : Tensor[B, 3n, 3n]
        """
        B, n, dev = x.shape[0], x.shape[1], x.device
        lb_vals = self.lb.expand(B, n)    # [B,n]
        ub_vals = self.ub.expand(B, n)    # [B,n]
        mask_lb = torch.isfinite(lb_vals).to(dev)         # [1,n,1] boolean
        mask_lb = mask_lb.unsqueeze(-1)                # [B,n,1]

        raw_x1 = (x - lb_vals)            # [B,n,1]
        x1     = torch.where(mask_lb, raw_x1,
                            torch.zeros_like(raw_x1))   
        
        mask_ub = torch.isfinite(ub_vals).to(dev)     # [1,n,1]
        mask_ub = mask_ub.unsqueeze(-1)              # [B,n,1]
        raw_x2 = (ub_vals - x)         # [B,n,1]
        x2     = torch.where(mask_ub, raw_x2,
                            torch.zeros_like(raw_x2))   

        # ------------------------------------------------------------------ #
        # gather blocks                                                       #
        # ------------------------------------------------------------------ #
        Q = self.Q.to(dev)
        Q = 0.5*(Q+Q.permute(0,2,1))                        # [B,n,n], SPD
        mu = muB.to(dev)                  # [B,1]

        z1_vec = z1.squeeze(-1)                       # [B,n]
        z2_vec = z2.squeeze(-1)                       # [B,n]

        # print(z1_vec.shape)
        # print(mu.shape)

        # d1_diag = 2.0 * mu / (z1_vec + mu)**2         # [B,n]
        # d2_diag = 2.0 * mu / (z2_vec + mu)**2         # [B,n]

        # D1 = torch.diag_embed(d1_diag)                # [B,n,n]
        # D2 = torch.diag_embed(d2_diag)                # [B,n,n]

        eps_div = 1e-20
        e_x1    = torch.ones_like(x1)                       
        inv_Z1mu = 1.0/(z1 + muB).clamp_min(eps_div)                      

        x1_mu = x1 + muB * e_x1                              

        Dz1   = torch.diag_embed(x1_mu * inv_Z1mu)  

        e_x2    = torch.ones_like(x2)                      
        inv_Z2mu = 1.0/(z2 + muB).clamp_min(eps_div)                          

        x2_mu = x2 + muB * e_x2                           

        Dz2   = torch.diag_embed(x2_mu * inv_Z2mu) 

        I   = torch.eye(n, device=dev).expand(B, -1, -1)

        eps = 1e-12
        s1p = (x1 + muB).clamp_min(eps)
        s2p = (x2 + muB).clamp_min(eps)
        t1p = (z1 + muB).clamp_min(eps)
        t2p = (z2 + muB).clamp_min(eps)

        c1 = z1E + x1E + muB
        c2 = z2E + x2E + muB

        # D1z = torch.diag_embed(s1p / t1p)
        # D2z = torch.diag_embed(s2p / t2p)

        Hxx = Q + 2*muB*torch.diag_embed(c1/(s1p**2) + c2/(s2p**2))

        print(Q)
        print(Dz1)
        print(Dz2)


        Ix = torch.eye(n, device=dev)
        I3 = torch.eye(3*n, device=dev)

        # Diagonals (batch means) to scale ridge magnitudes
        q_mean  = Hxx.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True)   # [B,1]
        d1_mean = Dz1.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True) # [B,1]
        d2_mean = Dz2.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True) # [B,1]

        # Choose small relative ridges
        lambda_x_rel = 1e-5
        lambda_z_rel = 1e-5  # even smaller; bump only if needed

        lam_x = (lambda_x_rel * (q_mean + 1.0)).view(B,1,1)   # [B,1,1]
        lam_z = (lambda_z_rel * ((d1_mean + d2_mean)/2 + 1.0)).view(B,1,1)

        # ------------------------------------------------------------------ #
        # build the full 3n × 3n Hessian for each batch                       #
        # ------------------------------------------------------------------ #
        top    = torch.cat([ Hxx ,   I,  -I], dim=2)     # [B,n,3n]
        middle = torch.cat([ I,  Dz1  ,   0*I], dim=2)   # [B,n,3n]
        bottom = torch.cat([-I,  0*I,  Dz2 ], dim=2)    # [B,n,3n]
        H      = torch.cat([top, middle, bottom], dim=1)   # [B,3n,3n]
        print("H: ", H.shape)

        # B, n, dev = x.shape[0], x.shape[1], x.device
        # m = 3 * n
        # I3 = torch.eye(m, device=dev, dtype=H.dtype).expand(B, -1, -1)

        # # ----- ridge (per batch) -----
        # # scale by average diagonal magnitude of blocks to be dimension-aware
        # qdiag = Q.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True)      # [B,1]
        # d1diag = Dz1.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True)   # [B,1]
        # d2diag = Dz2.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True)   # [B,1]
        # scale = (qdiag + d1diag + d2diag) / 3.0 + 1.0                               # avoid 0
        # ridge_rel = 1e-6                                                            # tune (1e-8..1e-3)
        # lam = (ridge_rel * scale).view(B, 1, 1)                                     # [B,1,1]

        # H_reg = H + lam * I3

        # ------------------------------------------------------------------ #
        # batched inverse  (n is tiny → explicit inverse is fine)             #
        # ------------------------------------------------------------------ #
        H_inv = torch.linalg.inv(H)                   # [B,3n,3n]
        print("H_inv: ", H_inv.shape)

        return H


    # def shifted_path_residual(self,
    #                           x,        # [B, n] or [B, n, 1]
    #                           s,        # [B, m] or [B, m, 1]
    #                           muP,      # scalar or [B,1,1]
    #                           muB):     # scalar or [B,1,1]
    #     """
    #     Build the “shifted path residual” vector for every batch item:
    #       • r_eq    = A x - b             ∈ ℝ^{[B, p, 1]}
    #       • r_ineq  = c(x) - s           ∈ ℝ^{[B, m, 1]}
    #       • r_comp  = s ∘ w - μ_B        ∈ ℝ^{[B, m, 1]}
        
    #     Here y = (c(x)-s)/μ_P,  w = μ_B/(s+μ_B) - y, and c(x)=ineq_resid(x).
    #     Finally, we concatenate [r_eq; r_ineq; r_comp] along the “constraint index” dimension
    #     (dim=1), producing a single tensor of shape [B, p + m + m, 1].  Taking its `norm()`
    #     gives a single ℓ₂‐residual per batch element.

    #     Returns
    #     -------
    #     r_all : torch.Tensor of shape [B, p + m + m, 1]
    #     """
    #     # 1)  Force x and s into column form [B,n,1] and [B,m,1]
    #     if x.dim() == 2:
    #         x_col = x.unsqueeze(-1)     # [B, n, 1]
    #     else:
    #         x_col = x                  # already [B, n, 1]

    #     if s.dim() == 2:
    #         s_col = s.unsqueeze(-1)     # [B, m, 1]
    #     else:
    #         s_col = s                  # already [B, m, 1]

    #     B = x_col.shape[0]   # batch size
    #     device = x_col.device

    #     # 2)  Equality‐constraint residual r_eq = A x - b  (if any)
    #     if self.num_eq > 0:
    #         # self.A is [B, p, n], x_col is [B, n, 1] → bmm → [B, p, 1]
    #         Ax = torch.bmm(self.A, x_col)         # [B, p, 1]
    #         r_eq = Ax - self.b                    # [B, p, 1]
    #     else:
    #         # no equalities → a “zero‐row” of size [B, 0, 1]
    #         r_eq = torch.zeros((B, 0, 1), device=device)

    #     # 3)  Inequality residual r_ineq = c(x) - s
    #     if self.num_ineq > 0:
    #         c_val = self.ineq_resid(x_col)        # [B, m, 1]
    #         r_ineq = c_val - s_col                # [B, m, 1]
    #     else:
    #         r_ineq = torch.zeros((B, 0, 1), device=device)
    #         c_val = r_ineq  # for the next step (so c_val always exists)

    #     # 4)  Reconstruct dual variables y and w:
    #     #       y = (c(x) - s) / μ_P
    #     #       w = μ_B / (s + μ_B)  -  y
    #     y = (c_val - s_col) / muP                 # [B, m, 1]
    #     w = muB / (s_col + muB) - y               # [B, m, 1]

    #     # 5)  Complementarity residual r_comp = s ∘ w - μ_B
    #     r_comp = s_col * w - muB                  # [B, m, 1]

    #     # 6)  Concatenate [r_eq; r_ineq; r_comp] along dim=1
    #     #     Final shape: [B, (p + m + m), 1]
    #     r_all = torch.cat([r_eq, r_ineq, r_comp], dim=1)

    #     return r_all
    
    def primal_feasibility(self,
            x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA):
        B, n, device = x.shape[0], x.shape[1], x.device
        x = x.unsqueeze(-1) if x.dim()==2 else x
        muB = muB.unsqueeze(-1)
        lb_vals = self.lb.expand(B, n)    # [B,n]
        ub_vals = self.ub.expand(B, n)    # [B,n]
        mask_lb = torch.isfinite(lb_vals).to(device)         # [1,n,1] boolean
        mask_lb = mask_lb.unsqueeze(-1)                # [B,n,1]

        residuals = []
        if self.num_eq != 0:
            res_eq = self.eq_resid(x)
            residuals.append(res_eq)

        if self.num_ineq != 0:
            c_x = self.ineq_resid(x)                  # [B, m_ineq, 1]
            s_slack = s.unsqueeze(-1) if s.dim()==2 else s
            residuals.append(c_x - s_slack)
            # s_neg = torch.minimum(s_slack, torch.zeros_like(s_slack))
            # residuals.append(s_neg)

        if self.num_var>0:
            # make everything [B,n,1]
            raw_x1 = (x - lb_vals.unsqueeze(-1)   )            # [B,n,1]
            x1     = torch.where(mask_lb, raw_x1,
                                torch.zeros_like(raw_x1))
            x1_slack = x1.unsqueeze(-1) if x1.dim()==2 else x1      # [B,n,1]
            lb = self.lb
            if lb.dim() == 2:                 # [B,n]
                lb = lb.unsqueeze(-1)    # → [B,n,1]
            scale_lb = torch.abs(lb) + 1.0
            x1_slack = x1_slack / scale_lb
            x1_violation = torch.minimum(x1_slack, torch.zeros_like(x1_slack))
            # lb_vals = self.lb
            # if lb_vals.dim() == 2:                 # [B,n]
            #     lb_vals = lb_vals.unsqueeze(-1)    # → [B,n,1]
            # mask_lb   = torch.isfinite(lb_vals).to(device)           # [1,n,1]
            # mask_lb   = mask_lb.expand(B, -1, -1)                    # [B,n,1]
            
            # # full residual
            # res_lb_full = x - lb_vals                     # [B,n,1]
            # # zero it out where lb was infinite
            # res_lb = torch.where(mask_lb, res_lb_full, 
            #                     torch.zeros_like(res_lb_full))
            #print(res_lb.mean().item())
            residuals.append(x1_violation)

        if self.num_var>0:
            mask_ub = torch.isfinite(ub_vals).to(device)     # [1,n,1]
            mask_ub = mask_ub.unsqueeze(-1)              # [B,n,1]
            raw_x2 = (ub_vals.unsqueeze(-1) - x)         # [B,n,1]
            x2     = torch.where(mask_ub, raw_x2,
                            torch.zeros_like(raw_x2))   
            x2_slack = x2.unsqueeze(-1) if x2.dim()==2 else x2      # [B,n,1]
            ub = self.ub
            if ub.dim() == 2:                 # [B,n]
                ub = ub.unsqueeze(-1)    # → [B,n,1]
            scale_ub = torch.abs(ub) + 1.0
            x2_slack = x2_slack / scale_ub
            x2_violation = torch.minimum(x2_slack, torch.zeros_like(x2_slack))
            # if ub_vals.dim() == 2:
            #     ub_vals = ub_vals.unsqueeze(-1)    # → [B,n,1]
            # mask_ub   = torch.isfinite(ub_vals).to(device)           # [1,n,1]
            # mask_ub   = mask_ub.expand(B, -1, -1)                    # [B,n,1]
            
            # res_ub_full = ub_vals - x                    # [B,n,1]
            # res_ub = torch.where(mask_ub, res_ub_full, 
            #                     torch.zeros_like(res_ub_full))
            #print(res_ub.mean().item())
            residuals.append(x2_violation)

        all_res = torch.cat(residuals, dim=1)

        return all_res.abs().amax(dim=(1,2)) 
    # change primal feasability 

    def dual_feasibility(self,
            x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA):
        B, n, device = x.shape[0], x.shape[1], x.device
        muB = muB.unsqueeze(-1)
        lb_vals = self.lb.expand(B, n)    # [B,n]
        ub_vals = self.ub.expand(B, n)    # [B,n]
        mask_lb = torch.isfinite(lb_vals).to(device)         # [1,n,1] boolean
        mask_lb = mask_lb.unsqueeze(-1)                # [B,n,1]
        mask_ub = torch.isfinite(ub_vals).to(device)     # [1,n,1]
        mask_ub = mask_ub.unsqueeze(-1)              # [B,n,1]

        # ensure x is [B,n,1]
        x_ = x.unsqueeze(-1) if x.dim()==2 else x    # [B,n,1]

        # 1) ∇f(x)
        res = self.obj_grad(x_)                       # [B,n,1]
        grad_norm = res.view(B, -1).norm(p=2, dim=1)    # ‖∇f‖₂ per batch
        print("g: ", grad_norm.mean().item())

        zScale = torch.maximum(torch.ones_like(grad_norm), grad_norm)  # [B]
        zScale = zScale.view(B, 1, 1)                     # make it broadcastable

        # 2) - A^T v
        if self.num_eq != 0:
            v_ = v.unsqueeze(-1) if v.dim()==2 else v # [B, neq,1]
            # expand A^T to [B,n,neq]
            A = self.A
            if A.dim() == 2:                           # shared (m_eq, n) → broadcast to batch
                A = A.unsqueeze(0).expand(B, -1, -1)
            At = A.transpose(1, 2)  
            res = res - torch.bmm(At, v_)              # [B,n,1]
            print("At: ", torch.bmm(At, v_).mean().item()  )

        # 3) - Jc(x)^T y
        if self.num_ineq != 0:
            y_  = y.unsqueeze(-1) if y.dim()==2 else y  # [B, mineq,1]
            Jg  = self.ineq_grad(x_)                    # [B,mineq,n]
            JgT = Jg.transpose(1,2)                     # [B,n,mineq]
            res = res - torch.bmm(JgT, y_)              # [B,n,1]
            print("JgT: ", torch.bmm(JgT, y_).mean().item() )

        # 4) - z1 only where lb > -∞
        if self.num_lb != 0:
            # bring lb to [1,n,1] and build mask
            raw_x1 = (x - lb_vals.unsqueeze(-1))            # [B,n,1]
            x1     = torch.where(mask_lb, raw_x1,
                                torch.zeros_like(raw_x1))
            if lb_vals.dim() == 2:                 # [B,n]
                lb_vals = lb_vals.unsqueeze(-1)    # → [B,n,1]

            z1_ = z1.unsqueeze(-1) if z1.dim()==2 else z1     # [B,n,1]
            res = res - torch.where(mask_lb, z1_,              # add z1_j if lb_j finite
                                    torch.zeros_like(z1_))

        # 5) + z2 only where ub < +∞
        if self.num_ub != 0:
            raw_x2 = (ub_vals.unsqueeze(-1) - x)         # [B,n,1]
            x2     = torch.where(mask_ub, raw_x2,
                                torch.zeros_like(raw_x2))   
            if ub_vals.dim() == 2:
                ub_vals = ub_vals.unsqueeze(-1)    # → [B,n,1]

            z2_ = z2.unsqueeze(-1) if z2.dim()==2 else z2     # [B,n,1]
            res = res + torch.where(mask_ub, z2_,              # subtract z2_j if ub_j finite
                                    torch.zeros_like(z2_))
        #print("res: ", res.mean().item())
            
        res_scaled = res / zScale

        #print("res: ", res.abs().amax(dim=(1, 2)).mean().item())

        # --------- build the bound–distance factors rL, rU ------------
        x1_slack = x1.unsqueeze(-1) if x1.dim() == 2 else x1   # [B,n,1]
        x2_slack = x2.unsqueeze(-1) if x2.dim() == 2 else x2

        if lb_vals.dim() == 2: lb_vals = lb_vals.unsqueeze(-1)  # [B,n,1]
        if ub_vals.dim() == 2: ub_vals = ub_vals.unsqueeze(-1)

        lb_vals = lb_vals.expand(B, -1, -1)      # broadcast
        ub_vals = ub_vals.expand(B, -1, -1)

        # r^L , r^U  (clamped to ≤1 later)
        rL = torch.zeros_like(x1_slack)
        rU = torch.zeros_like(x2_slack)
        rL[mask_lb] = x1_slack[mask_lb] / (torch.abs(lb_vals[mask_lb]) + 1.0)
        rU[mask_ub] = x2_slack[mask_ub] / (torch.abs(ub_vals[mask_ub]) + 1.0)

        # --------------- sign-filtered residuals -----------------------
        filtL = torch.minimum(rL, torch.ones_like(rL))      # min(rL,1)
        filtU = torch.minimum(rU, torch.ones_like(rU))

        zxL = torch.maximum(-res_scaled * filtL, torch.zeros_like(res_scaled))
        zxU = torch.maximum( res_scaled * filtU, torch.zeros_like(res_scaled))

        # ∞-norm dual residual ρ_dual  ----------------------------------
        all_dual = torch.cat([zxL, zxU], dim=1)   # [B, ≤2n,1]


        # print(torch.bmm(JgT, y_)[0])
        # print(torch.bmm(At, v_)[0])
        # print(self.obj_grad(x_)[0])
        # print(z1_[0])
        # print(z2_[0])

        return res_scaled.abs().amax(dim=(1, 2))    # shape [B]


    # def complementarity(self, x, x1, x2, s,       # add s1 and s2 if there are bounds for inequality
    #         y, v, z1, z2, w,    # add w1 and w2 instead of w if there are bounds for inequality
    #         x1E, x2E, sE, yE, vE, z1E, z2E, wE, # add w1E and w2E instead of wE if there are bounds for inequality
    #         muP, muB):
    #     B, device = x.shape[0], x.device
    #     residuals = []

    #     x_   = x.unsqueeze(-1) if x.dim() == 2 else x          # [B,n,1]
    #     muB = muB.unsqueeze(-1)
    #     grad = self.obj_grad(x_)                               # [B,n,1]
    #     grad_norm = grad.flatten(1).norm(p=2, dim=1)           # [B]
    #     zScale = torch.maximum(torch.ones_like(grad_norm), grad_norm)
    #     zScale = zScale.view(B, 1, 1)  

    #     # -- lower‐bound complementarity --
    #     if self.num_lb != 0:
    #         # shape [B,n,1]
    #         x1_slack = x1.unsqueeze(-1) if x1.dim()==2 else x1
    #         z1_slack = z1.unsqueeze(-1) if z1.dim()==2 else z1

    #         # mask only those vars with a finite lb
    #         lb_vals = self.lb
    #         if lb_vals.dim() == 2:                 # [B,n]
    #             lb_vals = lb_vals.unsqueeze(-1)    # → [B,n,1]
    #         mask_lb = torch.isfinite(lb_vals).to(device)        # [1,n,1]
    #         mask_lb = mask_lb.expand(B, -1, -1)                 # [B,n,1]

    #         # full complementarity residual
    #         res_lb_full = z1_slack * (x1_slack + muB) - muB     # [B,n,1]
    #         # zero out where lb = -∞
    #         res_lb = torch.where(mask_lb,
    #                             res_lb_full,
    #                             torch.zeros_like(res_lb_full))
    #         residuals.append(res_lb)

    #     # -- upper‐bound complementarity --
    #     if self.num_ub != 0:
    #         x2_slack = x2.unsqueeze(-1) if x2.dim()==2 else x2
    #         z2_slack = z2.unsqueeze(-1) if z2.dim()==2 else z2

    #         ub_vals = self.ub
    #         if ub_vals.dim() == 2:
    #             ub_vals = ub_vals.unsqueeze(-1)    # → [B,n,1]
    #         mask_ub = torch.isfinite(ub_vals).to(device)        # [1,n,1]
    #         mask_ub = mask_ub.expand(B, -1, -1)                 # [B,n,1]

    #         res_ub_full = z2_slack * (x2_slack + muB) - muB     # [B,n,1]
    #         res_ub = torch.where(mask_ub,
    #                             res_ub_full,
    #                             torch.zeros_like(res_ub_full))
    #         residuals.append(res_ub)

    #     # stack and return ∞-norm of all complementarity residuals
    #     #all_res = torch.cat(residuals, dim=1)                  # [B, ≤2n,1]
    #     if residuals:
    #         all_res = torch.cat(residuals, dim=1)                  # [B, ≤2n,1]
    #         scaled   = all_res / zScale                            # dimension-free
    #         return scaled.abs().amax(dim=(1, 2))                   # shape [B]
    #     else:
    #         # no finite bounds at all → residual identically zero
    #         return torch.zeros(B, device=device)

        # add the other blocks later
        #return all_res.abs().amax(dim=(1,2))  

    def complementarity(self,
            x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA):

        B, n, device = x.shape[0], x.shape[1], x.device
        lb_vals = self.lb.expand(B, n)    # [B,n]
        ub_vals = self.ub.expand(B, n)    # [B,n]
        mask_lb = torch.isfinite(lb_vals).to(device)         # [1,n,1] boolean
        mask_lb = mask_lb.unsqueeze(-1)                # [B,n,1]
        mask_ub = torch.isfinite(ub_vals).to(device)     # [1,n,1]
        mask_ub = mask_ub.unsqueeze(-1)              # [B,n,1]
        

        raw_x1 = (x - lb_vals.unsqueeze(-1))            # [B,n,1]
        x1     = torch.where(mask_lb, raw_x1,
                            torch.zeros_like(raw_x1))
        
        raw_x2 = (ub_vals.unsqueeze(-1) - x)         # [B,n,1]
        x2     = torch.where(mask_ub, raw_x2,
                            torch.zeros_like(raw_x2))   

        # ------------------------------------------------------------------
        # helper: broadcast everything to [B,n,1]
        # ------------------------------------------------------------------
        def to_col(t):
            return t.unsqueeze(-1) if t.dim() == 2 else t          # [B,n,1]

        x1 = to_col(x1);  z1 = to_col(z1)
        x2 = to_col(x2);  z2 = to_col(z2)
        muB = muB.unsqueeze(-1) if muB.dim() == 2 else muB         # [B,1,1]
        muB_x = muB.expand_as(x1)                                    # [B,n,1]
        zeros = torch.zeros_like(x1)

        # ------------------------------------------------------------------
        # lower-bound block  χ_comp^(L)
        # ------------------------------------------------------------------
        lb_mask = torch.isfinite(self.lb)
        if self.lb.dim() == 2: lb_mask = lb_mask                  # [B,n]
        lb_mask = lb_mask.unsqueeze(-1).to(device)                # [B,n,1]

        if lb_mask.any():                                         # at least one finite lb
            x1_mu = x1 + muB_x
            z1_mu = z1 + muB_x

            # q1(x1,z1) = max(|min(x1,z1,0)| , |x1 ∘ z1|)
            min_x1z1  = torch.min(torch.min(x1, z1), zeros)
            q1_L      = torch.maximum(min_x1z1.abs(), (x1 * z1).abs())

            # q2(x1,z1,μB) = max( μB e , |min(x1+μB, z1+μB, 0)| , |(x1+μB) ∘ (z1+μB)| )
            min_mu    = torch.min(torch.min(x1_mu, z1_mu), zeros)
            q2_L      = torch.maximum(
                            torch.maximum(muB, min_mu.abs()),
                            (x1_mu * z1_mu).abs()
                        )

            chi_L     = torch.minimum(q1_L, q2_L) * lb_mask       # zero where lb = −∞
        else:
            chi_L = zeros                                          # all zeros

        # ------------------------------------------------------------------
        # upper-bound block  χ_comp^(U)
        # ------------------------------------------------------------------
        ub_mask = torch.isfinite(self.ub)
        if self.ub.dim() == 2: ub_mask = ub_mask                   # [B,n]
        ub_mask = ub_mask.unsqueeze(-1).to(device)                 # [B,n,1]

        if ub_mask.any():                                          # at least one finite ub
            x2_mu = x2 + muB_x
            z2_mu = z2 + muB_x

            min_x2z2 = torch.min(torch.min(x2, z2), zeros)
            q1_U     = torch.maximum(min_x2z2.abs(), (x2 * z2).abs())

            min_mu2  = torch.min(torch.min(x2_mu, z2_mu), zeros)
            q2_U     = torch.maximum(
                            torch.maximum(muB, min_mu2.abs()),
                            (x2_mu * z2_mu).abs()
                        )

            chi_U    = torch.minimum(q1_U, q2_U) * ub_mask         # zero where ub = +∞
        else:
            chi_U = zeros

        if self.num_ineq > 0:
            s_   = to_col(s)                              # [B,m_ineq,1]
            w1_  = to_col(w1) if w1 is not None else torch.zeros_like(s_)
            muB_s = (muB if muB.dim()==3 else muB.unsqueeze(-1)).expand_as(s_)  # [B,m_ineq,1]
            zeros_m = torch.zeros_like(s_)

            # optional lower bound on s (defaults to 0)
            s_lb = getattr(self, 's_lb', torch.zeros_like(s_))
            mask_s_lb = torch.isfinite(s_lb)
            s1 = torch.where(mask_s_lb, s_ - s_lb, torch.zeros_like(s_))  # here s1 == s if lb=0

            s1_mu = s1 + muB_s
            w1_mu = w1_ + muB_s

            # q1(s1,w1) and q2(s1,w1, μB)
            min_sw   = torch.min(torch.min(s1, w1_), zeros_m)
            q1_SW    = torch.maximum(min_sw.abs(), (s1 * w1_).abs())
            min_muSW = torch.min(torch.min(s1_mu, w1_mu), zeros_m)
            q2_SW    = torch.maximum(torch.maximum(muB_s, min_muSW.abs()), (s1_mu * w1_mu).abs())

            chi_SW   = torch.minimum(q1_SW, q2_SW) * mask_s_lb  # (mask is all-ones if lb=0)

        # ------------------------------------------------------------------
        # scale by zScale = max{1, ||∇f(x)||₂}  (exactly like the code block)
        # ------------------------------------------------------------------
        grad     = self.obj_grad(to_col(x))                        # [B,n,1]
        grad2    = grad.flatten(1).norm(p=2, dim=1)                # [B]
        zScale   = torch.maximum(torch.ones_like(grad2), grad2)    # [B]
        zScale   = zScale.view(B, 1, 1)

        if self.num_ineq > 0:
            chi_tot  = torch.cat([chi_L, chi_U, chi_SW], dim=1) / zScale       # [B,≤2n,1]
        else:
            chi_tot  = torch.cat([chi_L, chi_U], dim=1) / zScale 
        return chi_tot.abs().amax(dim=(1, 2))                      # [B]


    
    def chi(self,
            x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA):
        """
        Composite residual χ = χ_feas + χ_stny + χ_comp
          χ_feas(v) = max_i |c_i(x) - s_i|
          χ_stny(v) = max_j ‖∇_x L_j(x,s,y,w)‖  (we approximate by your existing dual_feas here)
          χ_comp(v,μ) = max_i |s_i⋅w_i − μB|
        All return per‐instance [B] tensors; here we sum them.
        """
        # primal feasibility
        P = self.primal_feasibility(
            x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA)                 # [B]
        # stationarity (dual‐feas): use your existing D = dual_feasibility
        D = self.dual_feasibility(
            x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA)            # [B]
        # complementarity
        C = self.complementarity(
            x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA)               # [B]

        return P + D + C
    
    def Mtest(self,
            x, s,      
            y, v, z1, z2, w1, w2,    
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, 
            muP, muB, muA, M_max, bad_x1, bad_x2):
        """
        “M-iterate’’ consistency test – **box constraints only**.

        • Stationarity :  ‖∇f(x)+z1−z2‖∞
        • Lower path   :  ‖z1 − πL‖∞   with   πL = μB /(x1+μB)
        • Upper path   :  ‖z2 − πU‖∞   with   πU = μB /(x2+μB)
        • M-value      :  max(Mx,MzL,MzU) / max{1, |fM(x)|}
        """
        B, n, device = x.shape[0], x.shape[1], x.device
        lb_vals = self.lb.expand(B, n)    # [B,n]
        ub_vals = self.ub.expand(B, n)    # [B,n]
        mask_lb = torch.isfinite(lb_vals).to(device)         # [1,n,1] boolean
        mask_lb = mask_lb.unsqueeze(-1)                # [B,n,1]
        mask_ub = torch.isfinite(ub_vals).to(device)     # [1,n,1]
        mask_ub = mask_ub.unsqueeze(-1)              # [B,n,1]

        raw_x1 = (x - lb_vals.unsqueeze(-1))            # [B,n,1]
        x1     = torch.where(mask_lb, raw_x1,
                            torch.zeros_like(raw_x1))
        
        raw_x2 = (ub_vals.unsqueeze(-1) - x)         # [B,n,1]
        x2     = torch.where(mask_ub, raw_x2,
                            torch.zeros_like(raw_x2)) 

        grad_x, grad_s, grad_y, grad_v, grad_z1, grad_z2, grad_w1 = self.merit_grad_M(
            x, s,       # add s1 and s2 if there are bounds for inequality
            y, v, z1, z2, w1, w2,     # add w1 and w2 instead of w if there are bounds for inequality
            x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E, # add w1E and w2E instead of wE if there are bounds for inequality
            muP,
            muB, muA, bad_x1, bad_x2
        )

        M_x = grad_x.abs().amax(dim=(1, 2))  # [B]

        #M_x = torch.linalg.vector_norm(grad_x, dim=2).mean(dim=1) 

        eps = 1e-10

        # --- z1 (lower bounds) --------------------------------------------------
        DB1_diag = ((x1.squeeze(-1) + muB) / (z1.squeeze(-1) + muB)).abs().clamp_min(eps)  # D1^Z
        DB_norm_1 = DB1_diag.amax(dim=1)  # [B]
        M_z1 = grad_z1.abs().amax(dim=(1, 2)) / DB_norm_1  # [B]
        #M_z1 = torch.linalg.vector_norm(grad_z1, dim=2).mean(dim=1) / DB_norm_1 

        # --- z2 (upper bounds) --------------------------------------------------
        DB2_diag = ((x2.squeeze(-1) + muB) / (z2.squeeze(-1) + muB)).abs().clamp_min(eps)  # D2^Z
        DB_norm_2 = DB2_diag.amax(dim=1)  # [B]
        M_z2 = grad_z2.abs().amax(dim=(1, 2)) / DB_norm_2  # [B]
        #M_z2 = torch.linalg.vector_norm(grad_z2, dim=2).mean(dim=1) / DB_norm_2

        # --- w1 (inequality lower-bound slack) ---------------------------------
        if self.num_ineq != 0 and w1 is not None:
            M_s = grad_s.abs().amax(dim=(1, 2))  # [B]
            s_  = s if s.dim()  == 3 else s.unsqueeze(-1)    # [B,m_ineq,1]
            w1_ = w1 if w1.dim() == 3 else w1.unsqueeze(-1)  # [B,m_ineq,1]
            # s1 = s - s_lb (defaults to 0)
            s_lb = getattr(self, 's_lb', torch.zeros_like(s_))
            mask_s_lb = torch.isfinite(s_lb)
            s1 = torch.where(mask_s_lb, s_ - s_lb, torch.zeros_like(s_))

            DW1_diag = ((s1.squeeze(-1) + muB) / (w1_.squeeze(-1) + muB)).abs().clamp_min(eps)  # D1^W
            DW1_norm = DW1_diag.amax(dim=1)  # [B]
            M_w1 = grad_w1.abs().amax(dim=(1, 2)) / DW1_norm
            #M_w1 = torch.linalg.vector_norm(grad_w1, dim=2).mean(dim=1) / DW1_norm
        else:
            M_w1 = torch.zeros_like(M_x)
            M_s = torch.zeros_like(M_x)

        # --- y (inequality multipliers): scale by D_Y = μ^P I -------------------
        if self.num_ineq != 0 and grad_y is not None:
            DY_norm = muP.squeeze(-1).squeeze(-1).abs().clamp_min(eps)  # [B]
            M_y = grad_y.abs().amax(dim=(1, 2)) / DY_norm
            #M_y = torch.linalg.vector_norm(grad_y, dim=2).mean(dim=1) / DY_norm
        else:
            M_y = torch.zeros_like(M_x)

        # --- v (equality multipliers): scale by D_A = μ^A I ---------------------
        if self.num_eq != 0 and grad_v is not None:
            DA_norm = muA.squeeze(-1).squeeze(-1).abs().clamp_min(eps)  # [B]
            M_v = grad_v.abs().amax(dim=(1, 2)) / DA_norm
            #M_v = torch.linalg.vector_norm(grad_v, dim=2).mean(dim=1) / DA_norm
        else:
            M_v = torch.zeros_like(M_x)

        # ---------- aggregate & scale by initial merit --------------------------
        Mtest_value = torch.stack([M_x, M_s, M_z1, M_z2, M_w1, M_y, M_v], dim=1).amax(dim=1)  # [B]

        # #print("fM", fM.shape)
        fM0 = M_max.squeeze(-1).squeeze(-1)

        #print(M_max.shape)

        denom = torch.maximum(torch.ones_like(fM0), fM0)       # max{1, |fM|}
        return Mtest_value / denom                                    # [B]
    
    def primal_dual_infeasibility_45(
        self,
        x, s,
        y, v, z1, z2, w1, w2,
        x1E, x2E, s1E, s2E, yE, vE, z1E, z2E, w1E, w2E,
        muP, muB, muA,
    ):
        """
        Compute (e_P, e_D) as in equations (45a)-(45b) of the pdProj paper,
        extended to bound variables (x1,x2,z1,z2).

        Returns:
            eP: [B] primal infeasibility
            eD: [B] dual   infeasibility
        """
        device = x.device
        B, n, _ = x.shape

        # -------------------------
        # Helper: ∞-norm over (b, m, 1) or (b, n, 1)
        # -------------------------
        def inf_norm(t):
            # t: [B, m, 1] or [B, n, 1]
            return t.abs().amax(dim=(1, 2))

        # Keep running maxima for each batch element
        eP = torch.zeros(B, device=device)
        eD = torch.zeros(B, device=device)

        # ============================================================
        # 1) General inequalities: (s, w1, y) block
        # ============================================================
        if self.num_ineq > 0 and (s is not None):
            s_ = s if s.dim() == 3 else s.unsqueeze(-1)       # [B,m_ineq,1]
            y_ = y if (y is not None) else torch.zeros_like(s_)
            y_ = y_ if y_.dim() == 3 else y_.unsqueeze(-1)   # [B,m_ineq,1]
            w_ = w1 if (w1 is not None) else torch.zeros_like(s_)
            w_ = w_ if w_.dim() == 3 else w_.unsqueeze(-1)   # [B,m_ineq,1]

            # c(x) in the paper is our g(x) = Gx - c
            c_x = self.ineq_resid(x)                         # [B,m_ineq,1]

            # ---- Primal part: e_P^{ineq}
            # term A: min{0, s}
            min0s = torch.minimum(s_, torch.zeros_like(s_))
            termA = inf_norm(min0s)                          # [B]

            # term B: ||c(x) - s||_inf / max{1, ||s||_inf}
            s_inf = inf_norm(s_)                             # [B]
            scale_s = torch.maximum(torch.ones_like(s_inf), s_inf)
            termB = inf_norm(c_x - s_) / scale_s             # [B]

            eP_ineq = torch.maximum(termA, termB)            # [B]
            eP = torch.maximum(eP, eP_ineq)

            # ---- Dual part: e_D^{ineq}
            # grad f(x)
            grad_f = self.obj_grad(x)                        # [B,n,1]

            # Jacobian J(x) = G for linear constraints
            if hasattr(self, "G") and self.G is not None and self.num_ineq > 0:
                G = self.G.to(device)
                if G.dim() == 2:
                    G = G.unsqueeze(0).expand(B, -1, -1)     # [B,m_ineq,n]
                Jt_y = torch.bmm(G.transpose(1, 2), y_)      # [B,n,1]
                J_norm = G.abs().amax(dim=(1, 2))            # [B]
            else:
                Jt_y = torch.zeros_like(grad_f)
                J_norm = torch.zeros(B, device=device)

            grad_block = grad_f - Jt_y                       # [B,n,1]
            grad_block_inf = inf_norm(grad_block)            # [B]
            y_inf = inf_norm(y_)                             # [B]

            # σ = max{ 1, ||∇f(x)||, max{1,||y||} ||J|| }
            sigma = torch.maximum(
                torch.ones_like(grad_block_inf),
                torch.maximum(
                    grad_block_inf,
                    torch.maximum(torch.ones_like(y_inf), y_inf) * J_norm,
                ),
            )

            term1 = grad_block_inf / sigma                   # [B]
            term2 = inf_norm(w_ - y_)                        # [B]
            term3 = inf_norm(w_ * torch.minimum(torch.ones_like(s_), s_))

            eD_ineq = torch.max(torch.max(term1, term2), term3)
            eD = torch.maximum(eD, eD_ineq)

        # ============================================================
        # 2) Box lower bounds: x >= lb  → slacks x1, multipliers z1
        # ============================================================
        if self.num_lb != 0:
            lb = self.lb.to(device)
            if lb.dim() == 2:
                lb = lb.unsqueeze(-1)                        # [B,n,1] or [1,n,1]
            lb = lb.expand_as(x)                             # [B,n,1]

            # current lower slacks & mults
            # (we treat x1E as "current" slacks for this diagnostic)
            x1 = x1E if x1E is not None else (x - lb).clamp_min(0.0)
            x1 = x1 if x1.dim() == 3 else x1.unsqueeze(-1)   # [B,n,1]
            z1_ = z1 if (z1 is not None) else torch.zeros_like(x1)
            z1_ = z1_ if z1_.dim() == 3 else z1_.unsqueeze(-1)

            # primal: c^L(x) = x - lb
            cL = x - lb                                      # [B,n,1]

            min0x1 = torch.minimum(x1, torch.zeros_like(x1))
            termA_L = inf_norm(min0x1)

            x1_inf = inf_norm(x1)
            scale_x1 = torch.maximum(torch.ones_like(x1_inf), x1_inf)
            termB_L = inf_norm(cL - x1) / scale_x1

            eP_L = torch.maximum(termA_L, termB_L)
            eP = torch.maximum(eP, eP_L)

            # dual: no y for simple bounds; use z1 & complementarity
            term2_L = inf_norm(z1_)                          # like ||w||
            term3_L = inf_norm(z1_ * torch.minimum(torch.ones_like(x1), x1))
            eD_L = torch.maximum(term2_L, term3_L)
            eD = torch.maximum(eD, eD_L)

        # ============================================================
        # 3) Box upper bounds: x <= ub  → slacks x2, multipliers z2
        # ============================================================
        if self.num_ub != 0:
            ub = self.ub.to(device)
            if ub.dim() == 2:
                ub = ub.unsqueeze(-1)
            ub = ub.expand_as(x)                             # [B,n,1]

            x2 = x2E if x2E is not None else (ub - x).clamp_min(0.0)
            x2 = x2 if x2.dim() == 3 else x2.unsqueeze(-1)   # [B,n,1]
            z2_ = z2 if (z2 is not None) else torch.zeros_like(x2)
            z2_ = z2_ if z2_.dim() == 3 else z2_.unsqueeze(-1)

            cU = ub - x                                      # [B,n,1]

            min0x2 = torch.minimum(x2, torch.zeros_like(x2))
            termA_U = inf_norm(min0x2)

            x2_inf = inf_norm(x2)
            scale_x2 = torch.maximum(torch.ones_like(x2_inf), x2_inf)
            termB_U = inf_norm(cU - x2) / scale_x2

            eP_U = torch.maximum(termA_U, termB_U)
            eP = torch.maximum(eP, eP_U)

            term2_U = inf_norm(z2_)
            term3_U = inf_norm(z2_ * torch.minimum(torch.ones_like(x2), x2))
            eD_U = torch.maximum(term2_U, term3_U)
            eD = torch.maximum(eD, eD_U)

        return eP, eD







class convex_ipopt(ipopt.Problem):
    def __init__(self, Q, p, G, A, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = Q
        self.p = p
        self.G = G
        self.A = A
        if (self.G == 0.0).all():
            self.num_ineq = 0
        else:
            self.num_ineq = self.G.shape[0]
        if (self.A == 0.0).all():
            self.num_eq = 0
        else:
            self.num_eq = self.A.shape[0]

        self.objectives = []
        self.mus = []
        self.tril_indices = np.tril_indices(Q.shape[0])

    def objective(self, y):
        return 0.5 * (y @ self.Q @ y) + self.p@y

    def gradient(self, y):
        return 0.5*(self.Q+self.Q.T)@y + self.p

    def constraints(self, y):
        const_values = []
        if self.num_ineq != 0:
            const_values.append(self.G@y)
        if self.num_eq != 0:
            const_values.append(self.A@y)
        return np.hstack(const_values)

    def jacobian(self, y):
        const_jacob = []
        if self.num_ineq != 0:
            const_jacob.append(self.G.flatten())
        if self.num_eq != 0:
            const_jacob.append(self.A.flatten())
        return np.concatenate(const_jacob)

    def intermediate(self, alg_mod, iter_count, obj_value,
            inf_pr, inf_du, mu, d_norm, regularization_size,
            alpha_du, alpha_pr, ls_trials):
        self.objectives.append(obj_value)
        self.mus.append(mu)


class ConstrainedQP(ipopt.Problem):
    """
    minimize   0.5 x^T Q x + p^T x
    subject to lb <= x <= ub
               G x <= c          (inequality constraints, optional)
               A x  = b          (equality constraints, optional)
    """

    def __init__(self,
                 Q,
                 p,
                 lb,
                 ub,
                 G=None,
                 c=None,
                 A=None,
                 b=None,
                 tol=1e-2,
                 max_iter=500,
                 print_level=5):

        # --- basic shapes ---
        Q = np.asarray(Q, dtype=float)
        n = Q.shape[0]
        assert Q.shape == (n, n), "Q must be (n,n)"

        # symmetrize Q for safety
        self.Q = 0.5 * (Q + Q.T)

        p = np.asarray(p, dtype=float).reshape(-1)
        assert p.shape == (n,), "p must be length n"
        self.p = p

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        assert lb.shape == (n,) and ub.shape == (n,), "lb, ub must be length n"
        self.lb = lb
        self.ub = ub
        self.n = n

        # --- inequality constraints: G x <= c ---
        if G is None or (np.asarray(G).size == 1 and float(np.asarray(G)) == 0.0):
            self.num_ineq = 0
            self.G = np.zeros((0, n), dtype=float)
            self.c = None
        else:
            G = np.asarray(G, dtype=float)
            assert G.shape[1] == n, "G must be (m_ineq, n)"
            self.num_ineq = G.shape[0]
            self.G = G

            if c is None:
                raise ValueError("c must be provided when G is not None")
            c = np.asarray(c, dtype=float).reshape(self.num_ineq)
            self.c = c

        # --- equality constraints: A x = b ---
        if A is None or (np.asarray(A).size == 1 and float(np.asarray(A)) == 0.0):
            self.num_eq = 0
            self.A = np.zeros((0, n), dtype=float)
            self.b = None
        else:
            A = np.asarray(A, dtype=float)
            assert A.shape[1] == n, "A must be (m_eq, n)"
            self.num_eq = A.shape[0]
            self.A = A

            if b is None:
                raise ValueError("b must be provided when A is not None")
            b = np.asarray(b, dtype=float).reshape(self.num_eq)
            self.b = b

        m = self.num_ineq + self.num_eq

        # --- build Ipopt constraint bounds cl, cu ---
        # For Gx <= c:  -inf <= Gx <= c
        # For Ax  = b:   b  <= Ax <= b
        if self.num_ineq > 0:
            cl_ineq = -np.inf * np.ones(self.num_ineq, dtype=float)
            cu_ineq = self.c
        else:
            cl_ineq = np.empty(0, dtype=float)
            cu_ineq = np.empty(0, dtype=float)

        if self.num_eq > 0:
            cl_eq = self.b
            cu_eq = self.b
        else:
            cl_eq = np.empty(0, dtype=float)
            cu_eq = np.empty(0, dtype=float)

        if m > 0:
            cl = np.concatenate([cl_ineq, cl_eq])
            cu = np.concatenate([cu_ineq, cu_eq])
        else:
            cl = np.empty(0, dtype=float)
            cu = np.empty(0, dtype=float)

        # --- initialize Ipopt.Problem ---
        super().__init__(n=n, m=m, lb=lb, ub=ub, cl=cl, cu=cu)

        # Ipopt options
        self.add_option('tol', float(tol))
        self.add_option('max_iter', int(max_iter))
        self.add_option('print_level', int(print_level))
        self.add_option('hessian_approximation', 'limited-memory')

        # bookkeeping
        self.iters = 0
        self.objectives = []
        self.mus = []

    # ------------------------------------------------------------------
    # Required Ipopt callbacks
    # ------------------------------------------------------------------
    def objective(self, x):
        """
        f(x) = 0.5 x^T Q x + p^T x
        """
        return 0.5 * x.dot(self.Q.dot(x)) + self.p.dot(x)

    def gradient(self, x):
        """
        ∇f(x) = (Q + Q^T)/2 x + p = Q x + p  (since we symmetrized Q)
        """
        return self.Q.dot(x) + self.p

    def constraints(self, x):
        """
        Returns stacked constraints:
            [ Gx ; Ax ]
        Ipopt will enforce bounds cl <= constraints(x) <= cu
        """
        vals = []
        if self.num_ineq > 0:
            vals.append(self.G.dot(x))
        if self.num_eq > 0:
            vals.append(self.A.dot(x))

        if vals:
            return np.hstack(vals)
        else:
            return np.empty(0, dtype=float)

    def jacobian(self, x):
        """
        Dense Jacobian flattened in row-major order:
            [ G ;
              A ]
        """
        blocks = []
        if self.num_ineq > 0:
            blocks.append(self.G.flatten())
        if self.num_eq > 0:
            blocks.append(self.A.flatten())

        if not blocks:
            return np.empty(0, dtype=float)
        return np.concatenate(blocks)

    def jacobianstructure(self):
        """
        Dense structure: all entries of the (m x n) Jacobian are nonzero.
        """
        m = self.num_ineq + self.num_eq
        if m == 0:
            return (np.empty(0, dtype=int), np.empty(0, dtype=int))

        rows, cols = np.indices((m, self.n))
        return rows.flatten().astype(int), cols.flatten().astype(int)

    def intermediate(self, alg_mod, iter_count, obj_value,
                     inf_pr, inf_du, mu, d_norm, regularization_size,
                     alpha_du, alpha_pr, ls_trials):
        """
        Callback every Ipopt iteration; useful for logging.
        """
        self.iters = iter_count
        self.objectives.append(obj_value)
        self.mus.append(mu)
