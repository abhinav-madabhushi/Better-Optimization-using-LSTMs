import os
import osqp
import time
import torch
import numpy as np
import scipy.io as sio
import cyipopt as ipopt

from scipy.sparse import csc_matrix

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class QP_unconstrained(object):
    """
        minimize_x 0.5*x^T Q x + p^Tx
        s.t.       Gx <= c
                   Ax = b

        Q: [batch_size, num_var, num_var]
        p: [batch_size, num_var, 1]
        # G: [batch_size, num_ineq, num_var]
        # c: [batch_size, num_ineq, 1]
        # A: [batch_size, num_eq, num_var]
        # b: [batch_size, num_eq, 1]
    """
    def __init__(self, prob_type, learning_type, val_frac=0.0833, test_frac=0.0833, device='cpu', seed=17, **kwargs):
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
            self.data_size = data['Q'].shape[0]
            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size

            self.num_var = data['Q'].shape[1]
            # self.num_ineq = data['G'].shape[1]
            # self.num_eq = data['A'].shape[1]
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

            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size

            self.num_var = data['Q'].shape[1]
            try:
                self.num_ineq = data['G'].shape[1]
            except KeyError:
                self.num_ineq = 0

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
                self.Q = torch.tensor(data['Q'], device=self.device).float()[:self.train_size]
                self.p = torch.tensor(data['p'].astype(np.float32), device=self.device).float()[:self.train_size]
                if self.num_eq != 0:
                    self.A = torch.tensor(data['A'], device=self.device).float()[:self.train_size]
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
                self.Q = torch.tensor(data['Q'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                self.p = torch.tensor(data['p'].astype(np.float32), device=self.device).float()[self.train_size:self.train_size + self.val_size]
                if self.num_eq != 0:
                    self.A = torch.tensor(data['A'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
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
                self.Q = torch.tensor(data['Q'], device=self.device).float()[self.train_size + self.val_size:]
                self.p = torch.tensor(data['p'].astype(np.float32), device=self.device).float()[self.train_size + self.val_size:]
                if self.num_eq != 0:
                    self.A = torch.tensor(data['A'], device=self.device).float()[self.train_size + self.val_size:]
                    self.b = torch.tensor(data['b'].astype(np.float32), device=self.device).float()[self.train_size + self.val_size:]
                if self.num_ineq != 0:
                    self.G = torch.tensor(data['G'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                    self.c = torch.tensor(data['c'].astype(np.float32), device=self.device).float()[self.train_size:self.train_size + self.val_size]
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


    def obj_fn(self, x, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        return 0.5 * torch.bmm(x.permute(0, 2, 1), torch.bmm(Q, x)) + torch.bmm(p.permute(0, 2, 1), x)

    def obj_grad(self, x, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        return torch.bmm(0.5*(Q+Q.permute(0,2,1)), x) + p

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

    def opt_solve(self, solver_type='osqp', tol=1e-4, initial_y = None, init_mu=None, init_g=None, init_zl=None, init_zu=None):
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
        else:
            raise NotImplementedError
        
        return sols, total_time, parallel_time, np.array(iters).mean()
    
    def merit_M(self,
            x, s,       # primal x ∈ [B,n,1], slack s ∈ [B,m_ineq,1]
            y, w,       # duals y,w ∈ [B,m_ineq,1]
            sE_full, yE_full, wE_full, # shift estimates ∈ [B,m_ineq,1]
            muP, muB):  # scalars (float or [B,1,1])

        """
        Shifted penalty–barrier merit from Gill & Zhang (2023), Eq.(6),
        but with c(x) = [ g(x) ; A x - b ] in R^{m_tot}, and slacks for
        inequalities only (padded with zeros for equalities).
        Returns M_val ∈ [B,1,1].
        """

        # column‐form x and s
        x_col = x.unsqueeze(-1) if x.dim()==2 else x        # [B,n,1]
        s_full = s.unsqueeze(-1) if s.dim()==2 else s        # [B,m_tot,1]
        y_full = y.unsqueeze(-1) if y.dim()==2 else y        # [B,m_tot,1]
        w_full = w.unsqueeze(-1) if w.dim()==2 else w        # [B,m_tot,1]

        B, device = x_col.shape[0], x_col.device

        # building c(x) = [g(x); A x - b] : [B,m_tot,1]
        if self.num_ineq>0:
            g_col = self.ineq_resid(x_col)                  # [B,m_ineq,1]
        else:
            g_col = torch.zeros((B,0,1), device=device)
        if self.num_eq>0:
            eq_col = torch.bmm(self.A, x_col) - self.b      # [B,m_eq,1]
        else:
            eq_col = torch.zeros((B,0,1), device=device)
        c_val = torch.cat([g_col, eq_col], dim=1)           # [B,m_tot,1]

        # pad slacks/duals/shifts to length m_tot = m_ineq+m_eq
        m_eq = self.num_eq

        # primal objective
        f_val = self.obj_fn(x_col)                          # [B,1,1]

        diff = (c_val + s_full)

        # linear shift term
        lin_shift = torch.sum((diff)*yE_full,
                            dim=1, keepdim=True)        # [B,1,1]

        # quadratics
        quad1 = 0.5/muP * torch.sum(diff**2, dim=(1,2), keepdim=True)
        quad2 = 0.5/muP * torch.sum((diff + muP*(y_full - yE_full))**2,
                                    dim=(1,2), keepdim=True)

        # barriers
        alpha = wE_full + sE_full + muB                   # [B,m_tot,1]
        barr_s = -2*muB * torch.sum(alpha * torch.log(s_full + muB),
                                    dim=(1,2), keepdim=True)
        barr_w =   -muB * torch.sum(alpha * torch.log(w_full + muB),
                                    dim=(1,2), keepdim=True)

        # coupling and extra barrier
        coup  = torch.sum(w_full * (s_full + muB),
                        dim=(1,2), keepdim=True)
        extra = 2*muB * torch.sum(s_full,
                                dim=(1,2), keepdim=True)

        # assemble
        M_val = (f_val + lin_shift + quad1 + quad2 +
                barr_s + barr_w + coup + extra)          # [B,1,1]

        return M_val
    
    def merit_M_indi(self,
            x, s,       # primal x ∈ [B,n,1], slack s ∈ [B,m_ineq,1]
            y, w,       # duals y,w ∈ [B,m_ineq,1]
            sE_full, yE_full, wE_full, # shift estimates ∈ [B,m_ineq,1]
            muP, muB):  # scalars (float or [B,1,1])

        """
        Shifted penalty–barrier merit from Gill & Zhang (2023), Eq.(6),
        but with c(x) = [ g(x) ; A x - b ] in R^{m_tot}, and slacks for
        inequalities only (padded with zeros for equalities).
        Returns M_val ∈ [B,1,1].
        """

        # column‐form x and s
        x_col = x.unsqueeze(-1) if x.dim()==2 else x        # [B,n,1]
        s_full = s.unsqueeze(-1) if s.dim()==2 else s        # [B,m_ineq,1]
        y_full = y.unsqueeze(-1) if y.dim()==2 else y        # [B,m_ineq,1]
        w_full = w.unsqueeze(-1) if w.dim()==2 else w        # [B,m_ineq,1]

        B, device = x_col.shape[0], x_col.device

        # building c(x) = [g(x); A x - b] : [B,m_tot,1]
        if self.num_ineq>0:
            g_col = self.ineq_resid(x_col)                  # [B,m_ineq,1]
        else:
            g_col = torch.zeros((B,0,1), device=device)
        if self.num_eq>0:
            eq_col = torch.bmm(self.A, x_col) - self.b      # [B,m_eq,1]
        else:
            eq_col = torch.zeros((B,0,1), device=device)
        c_val = torch.cat([g_col, eq_col], dim=1)           # [B,m_tot,1]

        # pad slacks/duals/shifts to length m_tot = m_ineq+m_eq
        m_eq = self.num_eq

        # primal objective
        f_val = self.obj_fn(x_col)                          # [B,1,1]

        diff = (c_val + s_full)

        # linear shift term
        lin_shift = torch.sum((diff)*yE_full,
                            dim=1, keepdim=True)        # [B,1,1]

        # quadratics
        quad1 = 0.5/muP * torch.sum(diff**2, dim=(1,2), keepdim=True)
        quad2 = 0.5/muP * torch.sum((diff + muP*(y_full - yE_full))**2,
                                    dim=(1,2), keepdim=True)

        # barriers
        alpha = wE_full + sE_full + muB                   # [B,m_tot,1]
        barr_s = -2*muB * torch.sum(alpha * torch.log(s_full + muB),
                                    dim=(1,2), keepdim=True)
        barr_w =   -muB * torch.sum(alpha * torch.log(w_full + muB),
                                    dim=(1,2), keepdim=True)

        # coupling and extra barrier
        coup  = torch.sum(w_full * (s_full + muB),
                        dim=(1,2), keepdim=True)
        #print(coup.mean().item())
        extra = 2*muB * torch.sum(s_full,
                                dim=(1,2), keepdim=True)

        # assemble
        M_val = (f_val + lin_shift + quad1 + quad2 +
                barr_s + barr_w + coup + extra)          # [B,1,1]
        # print(f_val.mean().item() + lin_shift.mean().item() + quad1.mean().item() + quad2.mean().item() +
        #         barr_s.mean().item() + barr_w.mean().item() + coup.mean().item() + extra.mean().item() )

        return f_val, lin_shift, quad1, quad2, barr_s, barr_w, coup, extra
    


    def merit_grad_M_scaled(self,
                 x, s_full,         # [B,n] or [B,n,1],    [B,m_ineq] or [B,m_ineq,1]
                 y_full, w_full,         # [B,m_ineq] or [B,m_ineq,1]
                 sE_full, yE_full, wE_full,   # same shapes as s,y,w
                 muP, muB):    # scalar or [B,1,1]
        """
        ∇ of the shifted merit M(x,s,y,w) where
        c(x) = [ g(x) ; A x - b ] in R^{m_tot}.
        Returns grad_x:[B,n], grad_s,grad_y,grad_w:[B,m_tot].
        """
        x_col = x.unsqueeze(-1) if x.dim()==2 else x       # [B,n,1]

        B, device = x_col.shape[0], x_col.device
        n          = x_col.shape[1]
        m_ineq     = self.num_ineq
        m_eq       = self.num_eq
        m_tot      = m_ineq + m_eq

        # build stacked residual c_val = [g(x); A x - b]
        if m_ineq>0:
            g_col = self.ineq_resid(x_col)                 # [B,m_ineq,1]
            Jg    = self.ineq_grad(x_col)                  # [B,m_ineq,n]
        else:
            g_col = torch.zeros((B,0,1), device=device)
            Jg    = torch.zeros((B,0,n), device=device)

        if m_eq>0:
            eq_col = torch.bmm(self.A, x_col) - self.b     # [B,m_eq,1]
            Jeq    = self.A                                # [B,m_eq,n]
        else:
            eq_col = torch.zeros((B,0,1), device=device)
            Jeq    = torch.zeros((B,0,n), device=device)

        c_val = torch.cat([g_col, eq_col], dim=1)         # [B,m_tot,1]
        Jc    = torch.cat([Jg, Jeq],   dim=1)             # [B,m_tot,n]

        # flattening to 2D
        s_flat  = s_full.squeeze(-1)
        y_flat  = y_full.squeeze(-1)
        w_flat  = w_full.squeeze(-1)
        sE_flat = sE_full.squeeze(-1)
        yE_flat = yE_full.squeeze(-1)
        wE_flat = wE_full.squeeze(-1)
        d_flat  = (c_val + s_full).squeeze(-1)  # [B,m_tot]

        # grad_x = Qx + p  +  Jc^T [ 2 d/μP + (y-yE) - yE ]
        qx     = torch.bmm(self.Q, x_col).squeeze(-1)          # [B,n]
        p_flat = self.p.squeeze(-1).expand(B, -1)              # [B,n]
        grad_f = qx + p_flat                                   # [B,n]

        aux    = (2.0/muP.view(-1,1))*d_flat \
                + y_flat                                      # [B,m_tot]
        grad_c = torch.bmm(aux.unsqueeze(1), Jc).squeeze(1)    # [B,n]
        grad_x = grad_f + grad_c                               # [B,n]

        # grad_s = yE - 2 d/μP - (y-yE) - 2 μB α/(s+μB) + w + 2 μB
        alpha  = wE_flat + sE_flat + muB.view(-1,1)            # [B,m_tot]
        grad_s = ( 2.0*d_flat/muP.view(-1,1)
                + (y_flat)
                - 2.0*muB.view(-1,1)*alpha/(s_flat + muB.view(-1,1))
                + w_flat
                + 2.0*muB.view(-1,1)
                )                                        # [B,m_tot]

        # grad_y = d + μP(y - yE)
        grad_y = d_flat + muP.view(-1,1)*(y_flat - yE_flat)     # [B,m_tot]

        # grad_w = (s+μB) - μB α/(w+μB)
        grad_w = (s_flat + muB.view(-1,1)) \
                - muB.view(-1,1)*alpha/(w_flat + muB.view(-1,1)) # [B,m_tot]
        
        eps = 1e-8

        gx_norm = grad_x.norm(p=2, dim=1, keepdim=True)  # [B,1]
        grad_x = grad_x / (gx_norm + eps)

        gs_norm = grad_s.norm(p=2, dim=1, keepdim=True)
        grad_s = grad_s / (gs_norm + eps)

        gy_norm = grad_y.norm(p=2, dim=1, keepdim=True)
        grad_y = grad_y / (gy_norm + eps)

        gw_norm = grad_w.norm(p=2, dim=1, keepdim=True)
        grad_w = grad_w / (gw_norm + eps)

        return grad_x, grad_s, grad_y, grad_w
    
    def merit_grad_M(self,
                 x, s_full,         # [B,n] or [B,n,1],    [B,m_ineq] or [B,m_ineq,1]
                 y_full, w_full,         # [B,m_ineq] or [B,m_ineq,1]
                 sE_full, yE_full, wE_full,   # same shapes as s,y,w
                 muP, muB):    # scalar or [B,1,1]
        """
        ∇ of the shifted merit M(x,s,y,w) where
        c(x) = [ g(x) ; A x - b ] in R^{m_tot}.
        Returns grad_x:[B,n], grad_s,grad_y,grad_w:[B,m_tot].
        """
        x_col = x.unsqueeze(-1) if x.dim()==2 else x       # [B,n,1]

        B, device = x_col.shape[0], x_col.device
        n          = x_col.shape[1]
        m_ineq     = self.num_ineq
        m_eq       = self.num_eq
        m_tot      = m_ineq + m_eq

        # build stacked residual c_val = [g(x); A x - b]
        if m_ineq>0:
            g_col = self.ineq_resid(x_col)                 # [B,m_ineq,1]
            Jg    = self.ineq_grad(x_col)                  # [B,m_ineq,n]
        else:
            g_col = torch.zeros((B,0,1), device=device)
            Jg    = torch.zeros((B,0,n), device=device)

        if m_eq>0:
            eq_col = torch.bmm(self.A, x_col) - self.b     # [B,m_eq,1]
            Jeq    = self.A                                # [B,m_eq,n]
        else:
            eq_col = torch.zeros((B,0,1), device=device)
            Jeq    = torch.zeros((B,0,n), device=device)

        c_val = torch.cat([g_col, eq_col], dim=1)         # [B,m_tot,1]
        Jc    = torch.cat([Jg, Jeq],   dim=1)             # [B,m_tot,n]

        # flattening to 2D
        s_flat  = s_full.squeeze(-1)
        y_flat  = y_full.squeeze(-1)
        w_flat  = w_full.squeeze(-1)
        sE_flat = sE_full.squeeze(-1)
        yE_flat = yE_full.squeeze(-1)
        wE_flat = wE_full.squeeze(-1)
        d_flat  = (c_val + s_full).squeeze(-1)  # [B,m_tot]

        # grad_x = Qx + p  +  Jc^T [ 2 d/μP + (y-yE) - yE ]
        qx     = torch.bmm(self.Q, x_col).squeeze(-1)          # [B,n]
        p_flat = self.p.squeeze(-1).expand(B, -1)              # [B,n]
        grad_f = qx + p_flat                                   # [B,n]

        aux    = (2.0/muP.view(-1,1))*d_flat \
                + y_flat                                # [B,m_tot]
        grad_c = torch.bmm(aux.unsqueeze(1), Jc).squeeze(1)    # [B,n]
        grad_x = grad_f + grad_c                               # [B,n]

        # grad_s = yE - 2 d/μP - (y-yE) - 2 μB α/(s+μB) + w + 2 μB
        alpha  = wE_flat + sE_flat + muB.view(-1,1)            # [B,m_tot]
        grad_s = ( 2.0*d_flat/muP.view(-1,1)
                + (y_flat)
                - 2.0*muB.view(-1,1)*alpha/(s_flat + muB.view(-1,1))
                + w_flat
                + 2.0*muB.view(-1,1)
                )                                             # [B,m_tot]

        # grad_y = d + μP(y - yE)
        grad_y = d_flat + muP.view(-1,1)*(y_flat - yE_flat)     # [B,m_tot]

        # grad_w = (s+μB) - μB α/(w+μB)
        grad_w = (s_flat + muB.view(-1,1)) \
                - muB.view(-1,1)*alpha/(w_flat + muB.view(-1,1)) # [B,m_tot]

        return grad_x, grad_s, grad_y, grad_w

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
    
    def primal_feasibility(self, x_col, s_full):
        """
        Returns ∥[A x – b;  c(x) – s]∥_∞ over the batch.
        """
        B, device = x_col.shape[0], x_col.device
        # building c(x) = [g(x); A x - b] : [B,m_tot,1]
        if self.num_ineq>0:
            g_col = self.ineq_resid(x_col)                  # [B,m_ineq,1]
        else:
            g_col = torch.zeros((B,0,1), device=device)
        if self.num_eq>0:
            eq_col = torch.bmm(self.A, x_col) - self.b      # [B,m_eq,1]
        else:
            eq_col = torch.zeros((B,0,1), device=device)
        c_val = torch.cat([g_col, eq_col], dim=1)           # [B,m_tot,1]
        r_all = (c_val - s_full)
        return r_all.abs().amax(dim=(1,2))                # [B]

    def dual_feasibility(self, x_col, y_full, w_full):
        """
        Returns max( ∥∇f(x)+Jc(x)^T y∥_∞ , ∥y–w∥_∞ ) over the batch,
        where y_full and w_full are both [B, m_tot, 1], with the first
        num_ineq entries corresponding to Gx<=c duals and the last
        num_eq entries to Ax=b duals.
        """
        B, device = x_col.shape[0], x_col.device

        # 1) ∇f = Q x + p
        grad_f = (
            torch.bmm(self.Q, x_col).squeeze(-1)               # [B,n]
            + self.p.squeeze(-1).expand(B, -1)                 # [B,n]
        )

        # 2) split y_full into inequality & equality parts
        m_ineq, m_eq = self.num_ineq, self.num_eq
        y     = y_full.squeeze(-1)    # [B, m_tot]
        y_ineq = y[:, :m_ineq]        # [B, m_ineq]
        y_eq   = y[:, m_ineq:]        # [B, m_eq]

        # 3) build Jc^T y = G^T y_ineq + A^T y_eq
        grad_g = torch.zeros_like(grad_f)
        if m_ineq > 0:
            Jg     = self.ineq_grad(x_col)                   # [B, m_ineq, n]
            grad_g = torch.bmm(y_ineq.unsqueeze(1), Jg).squeeze(1)  # [B,n]

        grad_h = torch.zeros_like(grad_f)
        if m_eq > 0:
            # self.A is either [B, m_eq, n] or [m_eq, n] broadcasted
            A = self.A
            grad_h = torch.bmm(y_eq.unsqueeze(1), A).squeeze(1)     # [B,n]

        grad_L = grad_f + grad_g + grad_h                          # [B,n]

        # 4) ‖∇_xℒ‖_∞
        stat = grad_L.abs().amax(dim=1)                            # [B]

        # 5) ‖y – w‖_∞
        w = w_full.squeeze(-1)                                     # [B, m_tot]
        comp = (y - w).abs().amax(dim=1)                           # [B]

        # 6) take the maximum of the two
        return torch.max(stat, comp)                               # [B]


    def complementarity(self, s_full, w_full, muB):
        """
        PS-IPM complementarity residual:
        χ_comp(v,μB) = ‖min(q1,q2)‖_∞,
        where for each constraint i:
        q1_i = max( |min(s_i, w_i, 0)|, |s_i * w_i| )
        q2_i = max( μB, |min(s_i+μB, w_i+μB, 0)|, |(s_i+μB)*(w_i+μB)| )
        Inputs:
        s_col: [B, m_ineq, 1]
        w_col: [B, m_ineq, 1]
        muB:   scalar tensor or [B,1,1]
        Returns:
        χ_comp for each batch: shape [B]
        """
        B, device, dtype = s_full.shape[0], s_full.device, s_full.dtype
        m_ineq, m_eq = self.num_ineq, self.num_eq

        # 2) squeeze off the last dim → [B, m_tot]
        s = s_full.squeeze(-1)
        w = w_full.squeeze(-1)
        m_tot = m_ineq + m_eq

        # 3) broadcast μB to [B, m_tot]
        muB_flat = muB.view(-1,1)                   # [B,1]
        muB_mat  = muB_flat.expand(-1, m_tot)       # [B,m_tot]

        # 4) q1 = max( |min(s,w,0)|, |s*w| )
        min_sw0 = torch.min(torch.min(s, w), torch.zeros_like(s))
        q1      = torch.max(min_sw0.abs(), (s * w).abs())

        # 5) q2 = max( μB, |min(s+μB,w+μB,0)|, |(s+μB)*(w+μB)| )
        sB     = s + muB_mat
        wB     = w + muB_mat
        min_B0 = torch.min(torch.min(sB, wB), torch.zeros_like(sB))
        q2     = torch.max(
                    muB_mat,
                    torch.max(min_B0.abs(), (sB * wB).abs())
                )

        # 6) comp = min(q1, q2), then ∞-norm over constraints
        comp_vec = torch.min(q1, q2)              # [B, m_tot]
        return comp_vec.amax(dim=1)               # [B]

    
    def chi(self, x_col, s_full, y_full, w_full, muP, muB):
        """
        Composite residual χ = χ_feas + χ_stny + χ_comp
          χ_feas(v) = max_i |c_i(x) - s_i|
          χ_stny(v) = max_j ‖∇_x L_j(x,s,y,w)‖  (we approximate by your existing dual_feas here)
          χ_comp(v,μ) = max_i |s_i⋅w_i − μB|
        All return per‐instance [B] tensors; here we sum them.
        """
        # primal feasibility
        P = self.primal_feasibility(x_col, s_full)                 # [B]
        # stationarity (dual‐feas): use your existing D = dual_feasibility
        D = self.dual_feasibility(x_col, y_full, w_full)            # [B]
        # complementarity
        C = self.complementarity(s_full, w_full, muB)               # [B]

        return P + D + C





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
