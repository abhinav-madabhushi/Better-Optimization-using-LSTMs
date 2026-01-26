import os
import sys
import time
import torch
import numpy as np
import scipy.io as sio
import cyipopt as ipopt
from torch import Tensor

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Convex_QCQP(object):

    def __init__(self, prob_type, learning_type, val_frac=0.0833, test_frac=0.0833, device='cpu', seed=17, **kwargs):
        super().__init__()

        self.device = device
        self.seed = seed
        self.learning_type = learning_type
        self.train_frac = 1 - val_frac - test_frac
        self.val_frac = val_frac
        self.test_frac = test_frac


        if prob_type == 'Convex_QCQP_RHS':
            file_path = kwargs['file_path']
            data = sio.loadmat(file_path)
            self.data_size = data['X'].shape[0]

            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size
            torch.manual_seed(self.seed)

            self.num_var = data['Q'].shape[0]
            self.num_ineq = data['H'].shape[0]
            self.num_eq = data['A'].shape[0]
            self.num_lb = 0
            self.num_ub = 0

            if learning_type == 'train':
                self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.train_size, 1).unsqueeze(-1)
                self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
                self.b = torch.tensor(data['X'], device=self.device).float()[:self.train_size].unsqueeze(-1)
                self.Q_ineq = torch.tensor(data['H'], device=self.device).float()
                self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
                self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.train_size, 1).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'val':
                self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.val_size, 1).unsqueeze(-1)
                self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
                self.b = torch.tensor(data['X'], device=self.device).float()[self.train_size:self.train_size + self.val_size].unsqueeze(-1)
                self.Q_ineq = torch.tensor(data['H'], device=self.device).float()
                self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
                self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.val_size, 1).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'test':
                self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.test_size, 1).unsqueeze(-1)
                self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
                self.b = torch.tensor(data['X'], device=self.device).float()[self.train_size + self.val_size:].unsqueeze(-1)
                self.Q_ineq = torch.tensor(data['H'], device=self.device).float()
                self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
                self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.test_size, 1).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf

        elif prob_type == 'Convex_QCQP':
            self.data_size = kwargs['data_size']
            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size
            torch.manual_seed(self.seed)

            self.num_var = kwargs['num_var']
            self.num_ineq = kwargs['num_ineq']
            self.num_eq = kwargs['num_eq']
            self.num_lb = 0
            self.num_ub = 0

            if learning_type == 'train':
                self.Q = torch.diag_embed(0.5 * torch.rand(size=(self.data_size, self.num_var), device=device))[:self.train_size]
                self.p = 2 * torch.rand(size=(self.data_size, self.num_var), device=device)[:self.train_size].unsqueeze(-1) - 1
                self.A = 2 * torch.rand(size=(self.data_size, self.num_eq, self.num_var), device=self.device)[:self.train_size] - 1
                self.b = torch.rand(size=(self.data_size, self.num_eq, 1), device=self.device)[:self.train_size] - 0.5
                self.Q_ineq = torch.diag_embed(0.1 * torch.rand(size=(self.num_ineq, self.num_var), device=self.device))
                self.G = 2 * torch.rand(size=(self.data_size, self.num_ineq, self.num_var), device=device)[:self.train_size] - 1
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'val':
                self.Q = torch.diag_embed(0.5 * torch.rand(size=(self.data_size, self.num_var), device=device))[self.train_size:self.train_size + self.val_size]
                self.p = 2 * torch.rand(size=(self.data_size, self.num_var), device=device)[self.train_size:self.train_size + self.val_size].unsqueeze(-1) - 1
                self.A = 2 * torch.rand(size=(self.data_size, self.num_eq, self.num_var), device=self.device)[self.train_size:self.train_size + self.val_size] - 1
                self.b = torch.rand(size=(self.data_size, self.num_eq, 1), device=self.device)[self.train_size:self.train_size + self.val_size] - 0.5
                self.Q_ineq = torch.diag_embed(0.1 * torch.rand(size=(self.num_ineq, self.num_var), device=self.device))
                self.G = 2 * torch.rand(size=(self.data_size, self.num_ineq, self.num_var), device=device)[self.train_size:self.train_size + self.val_size] - 1
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'test':
                self.Q = torch.diag_embed(0.5 * torch.rand(size=(self.data_size, self.num_var), device=device))[self.train_size + self.val_size:]
                self.p = 2 * torch.rand(size=(self.data_size, self.num_var), device=device)[self.train_size + self.val_size:].unsqueeze(-1) - 1
                self.A = 2 * torch.rand(size=(self.data_size, self.num_eq, self.num_var), device=self.device)[self.train_size + self.val_size:] - 1
                self.b = torch.rand(size=(self.data_size, self.num_eq, 1), device=self.device)[self.train_size + self.val_size:] - 0.5
                self.Q_ineq = torch.diag_embed(0.1 * torch.rand(size=(self.num_ineq, self.num_var), device=self.device))
                self.G = 2 * torch.rand(size=(self.data_size, self.num_ineq, self.num_var), device=device)[self.train_size + self.val_size:] - 1
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf


    def obj_fn(self, x, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        return 0.5 * torch.bmm(x.permute(0, 2, 1), torch.bmm(Q, x)) + torch.bmm(p.permute(0, 2, 1), x)

    def obj_grad(self, x, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        return torch.bmm(0.5*(Q+Q.permute(0,2,1)), x) + p

    def ineq_resid(self, x, **kwargs):
        Q_ineq = kwargs.get('Q_ineq', self.Q_ineq)
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)

        res = []
        for i in range(self.num_ineq):
            resi = torch.bmm(x.permute(0, 2, 1) @ Q_ineq[i, :, :], x) + torch.bmm(G[:, i, :].unsqueeze(-1).permute(0, 2, 1), x) - c[:, i, :].unsqueeze(-1)
            res.append(resi)
        return torch.concat(res, dim=1)

    def ineq_dist(self, x, **kwargs):
        Q_ineq = kwargs.get('Q_ineq', self.Q_ineq)
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        return torch.clamp(self.ineq_resid(x, Q_ineq=Q_ineq, G=G, c=c), 0)

    def ineq_grad(self, x, **kwargs):
        Q_ineq = kwargs.get('Q_ineq', self.Q_ineq)
        G = kwargs.get('G', self.G)

        grad_list = []
        for i in range(self.num_ineq):
            grad_list.append(x.permute(0,2,1)@(Q_ineq[i,:,:].T+Q_ineq[i,:,:])+G[:,i,:].unsqueeze(1))
        return torch.concat(grad_list, dim=1)


    def eq_resid(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        return torch.bmm(A, x) - b

    def eq_dist(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        return torch.abs(self.eq_resid(x, A=A, b=b))

    def eq_grad(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        return A

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
            Q_ineq = kwargs.get('Q_ineq', self.Q_ineq)
            G = kwargs.get('G', self.G)
            c = kwargs.get('c', self.c)
        if self.num_eq != 0:
            A = kwargs.get('A', self.A)
            b = kwargs.get('b', self.b)
        if self.num_lb != 0:
            lb = kwargs.get('lb', self.lb)
        if self.num_ub != 0:
            ub = kwargs.get('ub', self.ub)
        batch_size = Q.shape[0]

        # residual
        F_list = []
        F1 = torch.bmm(0.5 * (Q + Q.permute(0, 2, 1)), x) + p
        if self.num_ineq != 0:
            F1 += torch.bmm(self.ineq_grad(x, Q_ineq=Q_ineq, G=G).permute(0, 2, 1), eta)
        if self.num_eq != 0:
            F1 += torch.bmm(self.eq_grad(x, A=A, b=b).permute(0, 2, 1), lamb)
        if self.num_lb != 0:
            F1 += -zl
        if self.num_ub != 0:
            F1 += zu
        F_list.append(F1)

        if self.num_ineq != 0:
            F2 = self.ineq_resid(x, Q_ineq=Q_ineq, G=G, c=c) + s
            F3 = eta * s
            F_list.append(F2)
            F_list.append(F3)

        if self.num_eq != 0:
            F4 = self.eq_resid(x, A=A, b=b)
            F_list.append(F4)

        if self.num_lb != 0:
            F5 = zl * (x - lb)
            F_list.append(F5)

        if self.num_ub != 0:
            F6 = zu * (ub - x)
            F_list.append(F6)

        F = torch.concat(F_list, dim=1)

        return F
    
    # def AT_lamb(self, x, eta, s, lamb, zl, zu, sigma, **kwargs):
    #     Q = kwargs.get('Q', self.Q)
    #     p = kwargs.get('p', self.p)
    #     mu = 0
    #     if self.num_ineq != 0:
    #         Q_ineq = kwargs.get('Q_ineq', self.Q_ineq)
    #         G = kwargs.get('G', self.G)
    #         c = kwargs.get('c', self.c)
    #         mu += sigma * ((eta * s).sum(1).unsqueeze(-1))
    #     if self.num_eq != 0:
    #         A = kwargs.get('A', self.A)
    #         b = kwargs.get('b', self.b)
    #     if self.num_lb != 0:
    #         lb = kwargs.get('lb', self.lb)
    #         mu += sigma * ((zl * (x - lb)).sum(1).unsqueeze(-1))
    #     if self.num_ub != 0:
    #         ub = kwargs.get('ub', self.ub)
    #         mu += sigma * ((zu * (ub - x)).sum(1).unsqueeze(-1))
    #     batch_size = Q.shape[0]
    #     # mu
    #     mu = mu / (self.num_ineq + self.num_lb + self.num_ub)
    #     AT_lamb = torch.bmm(self.eq_grad(x, A=A, b=b).permute(0,2,1), lamb)
    #     return (AT_lamb)
    
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
            Q_ineq = kwargs.get('Q_ineq', self.Q_ineq)
            G = kwargs.get('G', self.G)
            c = kwargs.get('c', self.c)
            mu += sigma * ((eta * s).sum(1).unsqueeze(-1))
        if self.num_eq != 0:
            A = kwargs.get('A', self.A)
            b = kwargs.get('b', self.b)
        if self.num_lb != 0:
            lb = kwargs.get('lb', self.lb)
            mu += sigma * ((zl * (x - lb)).sum(1).unsqueeze(-1))
        if self.num_ub != 0:
            ub = kwargs.get('ub', self.ub)
            mu += sigma * ((zu * (ub - x)).sum(1).unsqueeze(-1))
        batch_size = Q.shape[0]
        # mu
        mu = mu / (self.num_ineq + self.num_lb + self.num_ub)

        # residual
        F_list = []
        F1 = torch.bmm(0.5*(Q+Q.permute(0,2,1)), x) + p
        if self.num_ineq != 0:
            F1 += torch.bmm(self.ineq_grad(x, Q_ineq=Q_ineq, G=G).permute(0,2,1), eta)
        if self.num_eq != 0:
            F1 += torch.bmm(self.eq_grad(x, A=A, b=b).permute(0,2,1), lamb)
        if self.num_lb != 0:
            F1 += -zl
        if self.num_ub != 0:
            F1 += zu
        F_list.append(F1)

        if self.num_ineq != 0:
            F2 = self.ineq_resid(x, Q_ineq=Q_ineq, G=G, c=c) + s
            F3 = eta * s - mu
            F_list.append(F2)
            F_list.append(F3)

        if self.num_eq != 0:
            F4 = self.eq_resid(x, A=A, b=b)
            F_list.append(F4)

        if self.num_lb != 0:
            F5 = zl * (x - lb) - mu
            F_list.append(F5)

        if self.num_ub != 0:
            F6 = zu * (ub - x) - mu
            F_list.append(F6)

        F = torch.concat(F_list, dim=1)

        # jacobian of residual
        J_list = []
        J1 = 0.5*(Q+Q.permute(0,2,1))
        if self.num_ineq != 0:
            J1 = torch.concat((J1, self.ineq_grad(x, Q_ineq=Q_ineq, G=G).permute(0,2,1)), dim=2)
        if self.num_eq != 0:
            J1 = torch.concat((J1, self.eq_grad(x, A=A, b=b).permute(0,2,1)), dim=2)
        if self.num_ineq != 0:
            J1 = torch.concat((J1, torch.zeros(size=(batch_size, self.num_var, self.num_ineq), device=self.device)), dim=2)
        if self.num_lb != 0:
            J1 = torch.concat((J1, -torch.diag_embed(torch.ones(size=(batch_size, self.num_lb), device=self.device))), dim=2)
        if self.num_ub != 0:
            J1 = torch.concat((J1, torch.diag_embed(torch.ones(size=(batch_size, self.num_ub), device=self.device))), dim=2)
        J_list.append(J1)

        if self.num_ineq != 0:
            J2 = torch.concat((self.ineq_grad(x, Q_ineq=Q_ineq, G=G), torch.zeros(size=(batch_size, self.num_ineq, self.num_ineq), device=self.device)), dim=2)
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
            J4 = self.eq_grad(x, A=A, b=b)
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
            J5 = torch.concat((J5, torch.diag_embed((x - lb).squeeze(-1))), dim=2)
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
            J6 = torch.concat((J6, torch.diag_embed((ub - x).squeeze(-1))), dim=2)
            J_list.append(J6)

        J = torch.concat(J_list, dim=1)
        return J, F, mu

    def cal_kkt_newton(self, x, eta, s, lamb, zl, zu, mu, sigma, lambs, newton_flag = True, **kwargs):
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
        if self.num_ineq != 0:
            Q_ineq = kwargs.get('Q_ineq', self.Q_ineq)
            G = kwargs.get('G', self.G)
            c = kwargs.get('c', self.c)
            #mu += sigma * ((eta * s).sum(1).unsqueeze(-1))
        if self.num_eq != 0:
            A = kwargs.get('A', self.A)
            b = kwargs.get('b', self.b)
        if self.num_lb != 0:
            lb = kwargs.get('lb', self.lb)
            #mu += sigma * ((zl * (x - lb)).sum(1).unsqueeze(-1))
        if self.num_ub != 0:
            ub = kwargs.get('ub', self.ub)
            #mu += sigma * ((zu * (ub - x)).sum(1).unsqueeze(-1))
        batch_size = Q.shape[0]
        # mu
        # mu = mu / (self.num_ineq + self.num_lb + self.num_ub)

        # comp = 0.0                                              # tensor‑scalar

        # if self.num_ineq != 0:
        #     comp = comp + (eta * s).sum(dim=1, keepdim=True)    # ηᵀs

        # if self.num_lb != 0:
        #     comp = comp + (zl * (x - lb)).sum(dim=1, keepdim=True)

        # if self.num_ub != 0:
        #     comp = comp + (zu * (ub - x)).sum(dim=1, keepdim=True)

        # # 2.  Average over all complementarity pairs
        # denom = float(self.num_ineq + self.num_lb + self.num_ub)
        # mu    = comp / denom 
        # residual
        F_list = []
        F1 = torch.bmm(0.5*(Q+Q.permute(0,2,1)), x) + p
        if self.num_ineq != 0:
            F1 += torch.bmm(self.ineq_grad(x, Q_ineq=Q_ineq, G=G).permute(0,2,1), eta)
        if self.num_eq != 0:
            F1 += torch.bmm(self.eq_grad(x, A=A, b=b).permute(0,2,1), lamb)
        if self.num_lb != 0:
            F1 += -zl
        if self.num_ub != 0:
            F1 += zu
        F_list.append(F1)

        if self.num_ineq != 0:
            F2 = self.ineq_resid(x, Q_ineq=Q_ineq, G=G, c=c) + s
            F3 = eta * s - mu*sigma
            F_list.append(F2)
            F_list.append(F3)

        if self.num_eq != 0:
            F4 = self.eq_resid(x, A=A, b=b)
            F_list.append(F4)

        if self.num_lb != 0:
            F5 = zl * (x - lb) - mu*sigma
            F_list.append(F5)

        if self.num_ub != 0:
            F6 = zu * (ub - x) - mu*sigma
            F_list.append(F6)

        F = torch.concat(F_list, dim=1)

        # jacobian of residual
        J_list = []

        batch_size = Q.shape[0]
        n = Q.shape[1]

        # might need to update lambda and gamma based on some conditions
        H = 0.5 * (Q + Q.permute(0, 2, 1))                  # [B, n, n]
        
        I = torch.eye(n, device=H.device)              # [n, n]
        I = I.unsqueeze(0).repeat(batch_size, 1, 1)    # [B, n, n]
        # H = H + 1e-2*I
        # J1 = H
        if (newton_flag):
            H = H + lambs*I                             
            J1 = H
        else: 
            J1 = I
        # need to deal with lambs, reduce it in each iteration
        # 1a) compute the smallest eigenvalue of H (batch‑wise)
        # eigvals = torch.linalg.eigvalsh(H)           # [B, n]
        # min_eig  = eigvals.min(dim=1).values         # [B]

        # # 1b) compute a δ that just barely makes H PD
        # #    want H + δ I to have min_eig >= eps_tol
        # eps_tol = 1e-8
        # delta   = torch.clamp(eps_tol - min_eig, min=0.0)   # [B]
        # delta   = delta.unsqueeze(-1).unsqueeze(-1)         # [B,1,1]

        # # 1c) add it in
        # I       = torch.eye(n, device=H.device).unsqueeze(0).repeat(batch_size,1,1)
        # H       = H + (delta * I)


        # 1) Stationarity block ∂F1/∂[x, η, s, λ, zₗ, zᵤ]
        #J1 = 0.5 * (Q + Q.permute(0, 2, 1))  # ∂/∂x
        if self.num_ineq != 0:
            # η‑columns
            J1 = torch.concat((
                J1,
                self.ineq_grad(x, Q_ineq=Q_ineq, G=G).permute(0, 2, 1),
                # s‑columns (zero)
                torch.zeros((batch_size, self.num_var, self.num_ineq), device=self.device)
            ), dim=2)
        if self.num_eq != 0:
            # λ‑columns
            J1 = torch.concat((
                J1,
                self.eq_grad(x, A=A, b=b).permute(0, 2, 1)
            ), dim=2)
        if self.num_lb != 0:
            # zₗ‑columns
            J1 = torch.concat((
                J1,
                -torch.diag_embed(torch.ones((batch_size, self.num_lb), device=self.device))
            ), dim=2)
        if self.num_ub != 0:
            # zᵤ‑columns
            J1 = torch.concat((
                J1,
                torch.diag_embed(torch.ones((batch_size, self.num_ub), device=self.device))
            ), dim=2)
        J_list.append(J1)

        # 2) Inequality‐primal block ∂F2/∂[x, η, s, λ, zₗ, zᵤ]
        if self.num_ineq != 0:
            # ∂(g(x)+s)/∂x, η zeros, ∂/∂s, then λ, zₗ, zᵤ zeros
            J2 = torch.concat((
                self.ineq_grad(x, Q_ineq=Q_ineq, G=G),
                torch.zeros((batch_size, self.num_ineq, self.num_ineq), device=self.device),  # η
                torch.diag_embed(torch.ones((batch_size, self.num_ineq), device=self.device))  # s
            ), dim=2)
            if self.num_eq != 0:
                J2 = torch.concat((
                    J2,
                    torch.zeros((batch_size, self.num_ineq, self.num_eq), device=self.device)
                ), dim=2)
            if self.num_lb != 0:
                J2 = torch.concat((
                    J2,
                    torch.zeros((batch_size, self.num_ineq, self.num_lb), device=self.device)
                ), dim=2)
            if self.num_ub != 0:
                J2 = torch.concat((
                    J2,
                    torch.zeros((batch_size, self.num_ineq, self.num_ub), device=self.device)
                ), dim=2)
            J_list.append(J2)

        # 3) Complementarity η·s block ∂F3/∂[x, η, s, λ, zₗ, zᵤ]
        if self.num_ineq != 0:
            J3 = torch.zeros((batch_size, self.num_ineq, self.num_var), device=self.device)  # ∂/∂x
            J3 = torch.concat((
                J3,
                torch.diag_embed(s.squeeze(-1)),  # ∂(ηs)/∂η
                torch.zeros((batch_size, self.num_ineq, 0), device=self.device)  # placeholder if no λ
            ), dim=2) if self.num_eq == 0 else J3
            # better to do stepwise:
            J3 = torch.zeros((batch_size, self.num_ineq, self.num_var), device=self.device)
            J3 = torch.concat((
                J3,
                torch.diag_embed(s.squeeze(-1)),       # η
                torch.diag_embed(eta.squeeze(-1))      # s
            ), dim=2)
            # now pad λ, zₗ, zᵤ
            if self.num_eq != 0:
                J3 = torch.concat((
                    J3,
                    torch.zeros((batch_size, self.num_ineq, self.num_eq), device=self.device)
                ), dim=2)
            if self.num_lb != 0:
                J3 = torch.concat((
                    J3,
                    torch.zeros((batch_size, self.num_ineq, self.num_lb), device=self.device)
                ), dim=2)
            if self.num_ub != 0:
                J3 = torch.concat((
                    J3,
                    torch.zeros((batch_size, self.num_ineq, self.num_ub), device=self.device)
                ), dim=2)
            J_list.append(J3)

        # 4) Equality block ∂F4/∂[x, η, s, λ, zₗ, zᵤ]
        J4 = self.eq_grad(x, A=A, b=b)  # [B, num_eq, num_var]
         # pad η and s columns 
        if self.num_ineq != 0:
             J4 = torch.concat((
                 J4,
                 torch.zeros((batch_size, self.num_eq, self.num_ineq * 2), device=self.device)
             ), dim=2)
        # J4 = torch.concat((
        #      J4,
        #      torch.zeros((batch_size, self.num_eq, self.num_eq), device=self.device)
        # ), dim=2)

        gamma0 = 1e-1            # a “starting” scale
        gamma  = (gamma0 * mu)  # or make this an argument
        J4 = torch.concat((
            J4,
            -gamma * torch.eye(self.num_eq, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        ), dim=2)
         # pad any box‑constraint columns
        if self.num_lb != 0:
             J4 = torch.concat((
                 J4,
                 torch.zeros((batch_size, self.num_eq, self.num_lb), device=self.device)
            ), dim=2)
        if self.num_ub != 0:
             J4 = torch.concat((
                 J4,
                 torch.zeros((batch_size, self.num_eq, self.num_ub), device=self.device)
             ), dim=2)
        J_list.append(J4)

        # 5) Lower‐bound complementarity ∂F5/∂[x, η, s, λ, zₗ, zᵤ]
        if self.num_lb != 0:
            J5 = torch.diag_embed(zl.squeeze(-1))  # ∂(zₗ·(x−lb))/∂x
            # pad η,s,λ
            J5 = torch.concat((
                J5,
                torch.zeros((batch_size, self.num_lb, self.num_ineq*2 + self.num_eq), device=self.device)
            ), dim=2)
            # ∂/∂zₗ
            J5 = torch.concat((
                J5,
                torch.diag_embed((x - lb).squeeze(-1))
            ), dim=2)
            # pad zᵤ
            if self.num_ub != 0:
                J5 = torch.concat((
                    J5,
                    torch.zeros((batch_size, self.num_lb, self.num_ub), device=self.device)
                ), dim=2)
            J_list.append(J5)

        # 6) Upper‐bound complementarity ∂F6/∂[x, η, s, λ, zₗ, zᵤ]
        if self.num_ub != 0:
            J6 = -torch.diag_embed(zu.squeeze(-1))  # ∂(zᵤ·(ub−x))/∂x
            # pad η,s,λ,zₗ
            J6 = torch.concat((
                J6,
                torch.zeros((batch_size, self.num_ub, self.num_ineq*2 + self.num_eq + self.num_lb), device=self.device)
            ), dim=2)
            # ∂/∂zᵤ
            J6 = torch.concat((
                J6,
                torch.diag_embed((ub - x).squeeze(-1))
            ), dim=2)
            J_list.append(J6)

        # finally stack all row‐blocks
        J = torch.concat(J_list, dim=1)
        H = 0.5 * (Q + Q.permute(0, 2, 1))
        return J, F, H
    
    def cal_phi_grad(self, x, mu=1.0):
        """
        Compute the gradient and value of the log-barrier-augmented objective.

        Args:
            x: [B, num_var, 1] - current iterate
            mu: scalar - barrier parameter

        Returns:
            grad_phi: [B, num_var, 1] - gradient of barrier-augmented objective
            phi_val: [B, 1] - objective value
            dummy_mu: [B, 1, 1] - just for compatibility (zero-filled)
        """
        Q0 = self.Q
        p0 = self.p
        lb = self.lb
        ub = self.ub

        batch_size, n, _ = x.shape
        grad_phi = torch.bmm(Q0, x) + p0  # ∇f(x) = Qx + p
        phi_val = 0.5 * torch.bmm(x.transpose(1, 2), torch.bmm(Q0, x)).squeeze(-1) + torch.bmm(p0.transpose(1, 2), x).squeeze(-1)

        if self.num_ineq > 0:
            gx = self.ineq_resid(x)  # [B, m, 1], should be negative
            grad_gx = self.ineq_grad(x).permute(0, 2, 1)  # [B, n, m]
            grad_phi += mu * torch.bmm(grad_gx, 1.0 / gx)
            phi_val -= mu * torch.sum(torch.log(-gx), dim=1)

        if self.num_lb > 0:
            diff_lb = x - lb
            grad_phi += -mu / diff_lb
            phi_val -= mu * torch.sum(torch.log(diff_lb), dim=1)

        if self.num_ub > 0:
            diff_ub = ub - x
            grad_phi += mu / diff_ub
            phi_val -= mu * torch.sum(torch.log(diff_ub), dim=1)


        return grad_phi, phi_val

    def merit_phi_and_grads(self, x, s, *, mu, nu):
        """
        Args
        ----
        x : [B, n, 1]  – decision variables
        s : [B, m, 1]  – slack variables (must stay >0)
        mu: scalar     – barrier parameter
        nu: scalar     – ℓ₁ penalty weight for constraint violation

        Returns
        -------
        phi_val : Python float     =  mean_B  φ_ν(x,s)
        g_x     : [B, n, 1]        =  ∇ₓ φ_ν
        g_s     : [B, m, 1]        =  ∇ₛ φ_ν
        """
        B  = x.shape[0]
        eps = 1e-8                       # numerical safety

        # --- constraint residuals -----------------------------------------
        g_res  = self.ineq_resid(x)          # shape [B, m, 1]
        eq_res = self.eq_resid(x)            # [B, p, 1]   (p may be 0)
        dl_res = x - self.lb                 # [B, n, 1]   (−∞ if no bound)
        du_res = self.ub - x

        # --- objective & barrier pieces -----------------------------------
        f_val = self.obj_fn(x).squeeze(-1)   # [B]
        barrier  = -mu * torch.log(s+eps).sum(dim=(1,2))

        # --- ℓ₁ constraint‑violation penalty ------------------------------
        sign_g   = torch.sign(g_res + s)
        sign_eq  = torch.sign(eq_res)
        sign_dl  = torch.sign(dl_res)
        sign_du  = torch.sign(du_res)

        viol  = (torch.norm(g_res + s, p=1, dim=(1,2))
                + torch.norm(eq_res,       p=1, dim=(1,2))
                + torch.norm(dl_res,       p=1, dim=(1,2))
                + torch.norm(du_res,       p=1, dim=(1,2)))

        phi = f_val + barrier + nu * viol
        phi_val = phi.mean().item()

        # ---------- gradients ---------------------------------------------
        # ∇ₓ f
        g_x = torch.bmm(self.Q, x) + self.p                     # [B,n,1]

        # add ℓ₁ penalty contributions
        g_x += nu * ( torch.bmm(self.ineq_grad(x).permute(0,2,1), sign_g)
                    + torch.bmm(self.A.transpose(1,2),            sign_eq)
                    + sign_dl - sign_du )

        # ∇ₛ  φ_ν
        g_s = -mu / (s + eps) + nu * sign_g

        return phi_val, g_x, g_s


    def cal_kkt_jac_identity(self, x, eta, s, lamb, zl, zu, mu, sigma, **kwargs):
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
        if self.num_ineq != 0:
            Q_ineq = kwargs.get('Q_ineq', self.Q_ineq)
            G = kwargs.get('G', self.G)
            c = kwargs.get('c', self.c)
            #mu += sigma * ((eta * s).sum(1).unsqueeze(-1))
        if self.num_eq != 0:
            A = kwargs.get('A', self.A)
            b = kwargs.get('b', self.b)
        if self.num_lb != 0:
            lb = kwargs.get('lb', self.lb)
            #mu += sigma * ((zl * (x - lb)).sum(1).unsqueeze(-1))
        if self.num_ub != 0:
            ub = kwargs.get('ub', self.ub)
            #mu += sigma * ((zu * (ub - x)).sum(1).unsqueeze(-1))
        batch_size = Q.shape[0]
        # mu
        # mu = mu / (self.num_ineq + self.num_lb + self.num_ub)

        # comp = 0.0                                              # tensor‑scalar

        # if self.num_ineq != 0:
        #     comp = comp + (eta * s).sum(dim=1, keepdim=True)    # ηᵀs

        # if self.num_lb != 0:
        #     comp = comp + (zl * (x - lb)).sum(dim=1, keepdim=True)

        # if self.num_ub != 0:
        #     comp = comp + (zu * (ub - x)).sum(dim=1, keepdim=True)

        # # 2.  Average over all complementarity pairs
        # denom = float(self.num_ineq + self.num_lb + self.num_ub)
        # mu    = comp / denom 
        # residual
        F_list = []
        F1 = torch.bmm(0.5*(Q+Q.permute(0,2,1)), x) + p
        if self.num_ineq != 0:
            F1 += torch.bmm(self.ineq_grad(x, Q_ineq=Q_ineq, G=G).permute(0,2,1), eta)
        if self.num_eq != 0:
            F1 += torch.bmm(self.eq_grad(x, A=A, b=b).permute(0,2,1), lamb)
        if self.num_lb != 0:
            F1 += -zl
        if self.num_ub != 0:
            F1 += zu
        F_list.append(F1)

        if self.num_ineq != 0:
            F2 = self.ineq_resid(x, Q_ineq=Q_ineq, G=G, c=c) + s
            F3 = eta * s - sigma * mu
            F_list.append(F2)
            F_list.append(F3)

        if self.num_eq != 0:
            F4 = self.eq_resid(x, A=A, b=b)
            F_list.append(F4)

        if self.num_lb != 0:
            F5 = zl * (x - lb) - sigma * mu
            F_list.append(F5)

        if self.num_ub != 0:
            F6 = zu * (ub - x) - sigma * mu
            F_list.append(F6)

        F = torch.concat(F_list, dim=1)

        # jacobian of residual
        J_list = []

        # 1) Stationarity block ∂F1/∂[x, η, s, λ, zₗ, zᵤ]
        batch_size = x.shape[0]
        n = x.shape[1]   # number of decision variables

        # make a (batch_size × n × n) identity
        I = torch.eye(n, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)

        # use that in place of the Hessian
        J1 = I
        if self.num_ineq != 0:
            # η‑columns
            J1 = torch.concat((
                J1,
                self.ineq_grad(x, Q_ineq=Q_ineq, G=G).permute(0, 2, 1),
                # s‑columns (zero)
                torch.zeros((batch_size, self.num_var, self.num_ineq), device=self.device)
            ), dim=2)
        if self.num_eq != 0:
            # λ‑columns
            J1 = torch.concat((
                J1,
                self.eq_grad(x, A=A, b=b).permute(0, 2, 1)
            ), dim=2)
        if self.num_lb != 0:
            # zₗ‑columns
            J1 = torch.concat((
                J1,
                -torch.diag_embed(torch.ones((batch_size, self.num_lb), device=self.device))
            ), dim=2)
        if self.num_ub != 0:
            # zᵤ‑columns
            J1 = torch.concat((
                J1,
                torch.diag_embed(torch.ones((batch_size, self.num_ub), device=self.device))
            ), dim=2)
        J_list.append(J1)

        # 2) Inequality‐primal block ∂F2/∂[x, η, s, λ, zₗ, zᵤ]
        if self.num_ineq != 0:
            # ∂(g(x)+s)/∂x, η zeros, ∂/∂s, then λ, zₗ, zᵤ zeros
            J2 = torch.concat((
                self.ineq_grad(x, Q_ineq=Q_ineq, G=G),
                torch.zeros((batch_size, self.num_ineq, self.num_ineq), device=self.device),  # η
                torch.diag_embed(torch.ones((batch_size, self.num_ineq), device=self.device))  # s
            ), dim=2)
            if self.num_eq != 0:
                J2 = torch.concat((
                    J2,
                    torch.zeros((batch_size, self.num_ineq, self.num_eq), device=self.device)
                ), dim=2)
            if self.num_lb != 0:
                J2 = torch.concat((
                    J2,
                    torch.zeros((batch_size, self.num_ineq, self.num_lb), device=self.device)
                ), dim=2)
            if self.num_ub != 0:
                J2 = torch.concat((
                    J2,
                    torch.zeros((batch_size, self.num_ineq, self.num_ub), device=self.device)
                ), dim=2)
            J_list.append(J2)

        # 3) Complementarity η·s block ∂F3/∂[x, η, s, λ, zₗ, zᵤ]
        if self.num_ineq != 0:
            J3 = torch.zeros((batch_size, self.num_ineq, self.num_var), device=self.device)  # ∂/∂x
            J3 = torch.concat((
                J3,
                torch.diag_embed(s.squeeze(-1)),  # ∂(ηs)/∂η
                torch.zeros((batch_size, self.num_ineq, 0), device=self.device)  # placeholder if no λ
            ), dim=2) if self.num_eq == 0 else J3
            # better to do stepwise:
            J3 = torch.zeros((batch_size, self.num_ineq, self.num_var), device=self.device)
            J3 = torch.concat((
                J3,
                torch.diag_embed(s.squeeze(-1)),       # η
                torch.diag_embed(eta.squeeze(-1))      # s
            ), dim=2)
            # now pad λ, zₗ, zᵤ
            if self.num_eq != 0:
                J3 = torch.concat((
                    J3,
                    torch.zeros((batch_size, self.num_ineq, self.num_eq), device=self.device)
                ), dim=2)
            if self.num_lb != 0:
                J3 = torch.concat((
                    J3,
                    torch.zeros((batch_size, self.num_ineq, self.num_lb), device=self.device)
                ), dim=2)
            if self.num_ub != 0:
                J3 = torch.concat((
                    J3,
                    torch.zeros((batch_size, self.num_ineq, self.num_ub), device=self.device)
                ), dim=2)
            J_list.append(J3)

        # 4) Equality block ∂F4/∂[x, η, s, λ, zₗ, zᵤ]
        J4 = self.eq_grad(x, A=A, b=b)  # [B, num_eq, num_var]
         # pad η and s columns 
        if self.num_ineq != 0:
             J4 = torch.concat((
                 J4,
                 torch.zeros((batch_size, self.num_eq, self.num_ineq * 2), device=self.device)
             ), dim=2)
         # --- pad the λ‑columns (you were missing this!) ---
        J4 = torch.concat((
             J4,
             torch.zeros((batch_size, self.num_eq, self.num_eq), device=self.device)
        ), dim=2)
         # pad any box‑constraint columns
        if self.num_lb != 0:
             J4 = torch.concat((
                 J4,
                 torch.zeros((batch_size, self.num_eq, self.num_lb), device=self.device)
            ), dim=2)
        if self.num_ub != 0:
             J4 = torch.concat((
                 J4,
                 torch.zeros((batch_size, self.num_eq, self.num_ub), device=self.device)
             ), dim=2)
        J_list.append(J4)

        # 5) Lower‐bound complementarity ∂F5/∂[x, η, s, λ, zₗ, zᵤ]
        if self.num_lb != 0:
            J5 = torch.diag_embed(zl.squeeze(-1))  # ∂(zₗ·(x−lb))/∂x
            # pad η,s,λ
            J5 = torch.concat((
                J5,
                torch.zeros((batch_size, self.num_lb, self.num_ineq*2 + self.num_eq), device=self.device)
            ), dim=2)
            # ∂/∂zₗ
            J5 = torch.concat((
                J5,
                torch.diag_embed((x - lb).squeeze(-1))
            ), dim=2)
            # pad zᵤ
            if self.num_ub != 0:
                J5 = torch.concat((
                    J5,
                    torch.zeros((batch_size, self.num_lb, self.num_ub), device=self.device)
                ), dim=2)
            J_list.append(J5)

        # 6) Upper‐bound complementarity ∂F6/∂[x, η, s, λ, zₗ, zᵤ]
        if self.num_ub != 0:
            J6 = -torch.diag_embed(zu.squeeze(-1))  # ∂(zᵤ·(ub−x))/∂x
            # pad η,s,λ,zₗ
            J6 = torch.concat((
                J6,
                torch.zeros((batch_size, self.num_ub, self.num_ineq*2 + self.num_eq + self.num_lb), device=self.device)
            ), dim=2)
            # ∂/∂zᵤ
            J6 = torch.concat((
                J6,
                torch.diag_embed((ub - x).squeeze(-1))
            ), dim=2)
            J_list.append(J6)

        # finally stack all row‐blocks
        J = torch.concat(J_list, dim=1)
        return J, F
    
    def project_onto_Omega(self, v, bL, bU, muB, sigma):
        """
        Project the combined variable vector v onto the perturbed box
        [bL + sigma*muB, bU - sigma*muB] componentwise.
        """
        lower = bL + sigma * muB
        upper = bU - sigma * muB
        return torch.max(torch.min(v, upper), lower)

    def merit(self, x, eta, s, lamb, yE, sE, wE, muP, muB):
        """
        All-shifted penalty-barrier M(x,s,y,w ; sE,yE,wE,muP,muB), Eq. (6) of the paper.
        """
        # f(x)
        fval = self.obj_fn(x)

        # (B) linear shift term
        if self.num_ineq > 0:
            cval = self.ineq_resid(x)    # c(x)
        else:
            cval = torch.zeros_like(s)
        lin_shift = - torch.sum((cval - s) * yE, dim=1, keepdim=True)

        # (C)+(D) quadratic penalties
        p1 = 0.5/muP * torch.sum((cval - s)**2, dim=1, keepdim=True)
        p2 = 0.5/muP * torch.sum((cval - s + muP*(eta - yE))**2,
                                dim=1, keepdim=True)

        # (E)+(F) barriers on s and eta with combined shifts
        wE_sE = wE + sE + muB
        barr_s = - torch.sum( muB * wE_sE * torch.log(s + muB),
                              dim=1, keepdim=True)
        barr_eta = - torch.sum( muB * wE_sE * torch.log(eta + muB),
                                dim=1, keepdim=True)

        # (G)+(H) linear terms
        lin_sw = torch.sum(eta*(s + muB), dim=1, keepdim=True) \
               + 2*muB*torch.sum(s, dim=1, keepdim=True)

        return fval + lin_shift + p1 + p2 + barr_s + barr_eta + lin_sw
    
    def merit_grad(self, x, eta, s, lamb, yE, sE, wE, muP, muB):
        """
        Returns ∇M stacked as [∂/∂x; ∂/∂η; ∂/∂s; ∂/∂λ; ∂/∂zₗ; ∂/∂zᵤ],
        matching sizes [n, m, m, p, nₗ, nᵤ].
        """
        B = x.shape[0]
        # --- ∂/∂x ---
        grad_x = self.obj_grad(x)                               # [B, n, 1]

        if self.num_ineq > 0:
            cval  = self.ineq_resid(x)                          # [B, m, 1]
            Jc    = self.ineq_grad(x)                           # [B, m, n]
            wEsE  = wE + sE + muB                              # [B, m, 1]

            # linear + quadratic terms
            term = (2*(cval - s) + muP*(eta - yE))/muP - yE     # [B, m,1]
            grad_x = grad_x + torch.bmm(Jc.permute(0,2,1), term)  # [B, n,1]

            # ∂/∂η
            grad_eta = (cval - s + muP*(eta - yE))              # p2
            grad_eta = grad_eta + (s + muB)                     # +η(s+μB)
            grad_eta = grad_eta - muB * wEsE/(eta + muB)        # barrier on η

            # ∂/∂s
            grad_s  = yE                                       # from –(c–s)·yE
            grad_s  = grad_s - (cval - s)/muP                  # p1
            grad_s  = grad_s - (cval - s + muP*(eta - yE))/muP  # p2
            grad_s  = grad_s + eta                             # +η(s+μB)
            grad_s  = grad_s + 2*muB                           # +2μB·s
            grad_s  = grad_s - muB * wEsE/(s + muB)             # barrier on s
        else:
            grad_eta = torch.zeros((B, self.num_ineq, 1), device=x.device)
            grad_s   = torch.zeros((B, self.num_ineq, 1), device=x.device)

        # ∂/∂λ = 0
        grad_lamb = torch.zeros((B, self.num_eq, 1), device=x.device)
        # ∂/∂zₗ = 0 of size nₗ
        grad_zl   = torch.zeros((B, self.num_lb, 1), device=x.device)
        # ∂/∂zᵤ = 0 of size nᵤ
        grad_zu   = torch.zeros((B, self.num_ub, 1), device=x.device)

        return torch.cat([grad_x,
                          grad_eta,
                          grad_s,
                          grad_lamb,
                          grad_zl,
                          grad_zu], dim=1)


    def calculate_F(self, x, eta, s, lamb, zl, zu, sigma, **kwargs):
        """
        x:    [batch_size, num_var, 1]
        eta:  [batch_size, num_ineq, 1]
        s:    [batch_size, num_ineq, 1]
        lamb: [batch_size, num_eq,   1]
        zl:   [batch_size, num_lb,   1]
        zu:   [batch_size, num_ub,   1]
        sigma: float

        Returns:
        --------
        F:   [batch_size, total_dim, 1]  # The KKT residual vector
        mu:  [batch_size, 1, 1]          # The barrier (or central path) parameter
        """

        # Pull out relevant data (or default to self fields)
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)

        # For inequality constraints, get relevant data
        mu = 0.0
        if self.num_ineq != 0:
            Q_ineq = kwargs.get('Q_ineq', self.Q_ineq)  # if your code uses Q_ineq
            G = kwargs.get('G', self.G)
            c = kwargs.get('c', self.c)
            # Add the sigma*(eta*s) portion for mu
            mu += sigma * ((eta * s).sum(1).unsqueeze(-1))

        # For equality constraints
        if self.num_eq != 0:
            A = kwargs.get('A', self.A)
            b = kwargs.get('b', self.b)

        # For lower bounds
        if self.num_lb != 0:
            lb = kwargs.get('lb', self.lb)
            mu += sigma * ((zl * (x - lb)).sum(1).unsqueeze(-1))

        # For upper bounds
        if self.num_ub != 0:
            ub = kwargs.get('ub', self.ub)
            mu += sigma * ((zu * (ub - x)).sum(1).unsqueeze(-1))

        # Compute average mu if you have multiple constraints/bounds
        denom = (self.num_ineq + self.num_lb + self.num_ub)
        if denom > 0:
            mu = mu / denom

        batch_size = Q.shape[0]

        # Build up the blocks of F in a list
        F_list = []

        # 1) Stationarity part: 0.5*(Q + Q^T)x + p + ...
        F1 = torch.bmm(0.5 * (Q + Q.permute(0, 2, 1)), x) + p

        # If you have inequality gradient terms (like G^T eta in a simple QP),
        # your code might do something like self.ineq_grad(...).  We replicate that:
        if self.num_ineq != 0:
            # The function self.ineq_grad(...) presumably returns G or Q_ineq-based partial derivatives.
            F1 += torch.bmm(self.ineq_grad(x, Q_ineq=Q_ineq, G=G).permute(0, 2, 1), eta)

        # For equality constraints
        if self.num_eq != 0:
            F1 += torch.bmm(self.eq_grad(x, A=A, b=b).permute(0, 2, 1), lamb)

        # Lower bound terms: -zl
        if self.num_lb != 0:
            F1 += -zl

        # Upper bound terms: +zu
        if self.num_ub != 0:
            F1 += zu

        F_list.append(F1)

        # 2) In equality constraints, if num_ineq > 0, build F2 (the primal feasibility for Gx - c + s = 0)
        #    and F3 (the complementarity eta*s - mu = 0)
        if self.num_ineq != 0:
            F2 = self.ineq_resid(x, Q_ineq=Q_ineq, G=G, c=c) + s
            F3 = eta * s - mu
            F_list.append(F2)
            F_list.append(F3)

        # 3) For equality constraints: Ax - b = 0
        if self.num_eq != 0:
            F4 = self.eq_resid(x, A=A, b=b)
            F_list.append(F4)

        # 4) For lower bounds: zl*(x - lb) - mu = 0
        if self.num_lb != 0:
            F5 = zl * (x - lb) - mu
            F_list.append(F5)

        # 5) For upper bounds: zu*(ub - x) - mu = 0
        if self.num_ub != 0:
            F6 = zu * (ub - x) - mu
            F_list.append(F6)

        # Concatenate everything along dim=1
        F = torch.cat(F_list, dim=1)

        return F, mu

    def compute_full_derivatives(self, y, sigma, **kwargs):
        """
        Compute the full Hessian H of the merit function ϕ(y) = 0.5 * ||F(y)||^2,
        along with the Jacobian J of F and F itself.
        
        Input:
          y: a tensor of shape (total_vars, 1) containing the concatenated variables
             [x; eta; s; lamb]. Make sure y.requires_grad is True.
          sigma: the sigma parameter (passed to calculate_F)
          kwargs: any additional keyword arguments for calculate_F
          
        Returns:
          H: the Hessian of ϕ(y) with respect to y (shape (total_vars, total_vars))
          J: the Jacobian of F with respect to y (shape (total_vars, total_vars))
          F: the KKT residual vector evaluated at y (shape (total_vars,))
          
        Note: This function is written for a single instance (i.e. batch_size=1).
        """
        total_vars = y.shape[0]
        num_var   = self.num_var
        num_ineq  = self.num_ineq
        num_eq    = self.num_eq

        # Define a function that maps y (flattened) to the merit function phi.
        def phi_func(y_flat):
            # y_flat is of shape (total_vars,)
            y_in = y_flat.view(total_vars, 1)
            # Extract components:
            x    = y_in[:num_var].unsqueeze(0)  # shape: (1, num_var, 1)
            if num_ineq > 0:
                eta  = y_in[num_var:num_var+num_ineq].unsqueeze(0)  # (1, num_ineq, 1)
                s    = y_in[num_var+num_ineq:num_var+2*num_ineq].unsqueeze(0)
            else:
                eta = torch.empty((1, 0, 1), device=y.device)
                s   = torch.empty((1, 0, 1), device=y.device)
            if num_eq > 0:
                lamb = y_in[num_var+2*num_ineq:num_var+2*num_ineq+num_eq].unsqueeze(0)
            else:
                lamb = torch.empty((1, 0, 1), device=y.device)
            # Compute F using your calculate_F method.
            F_val, _ = self.calculate_F(x, eta, s, lamb, None, None, sigma, **kwargs)
            # F_val has shape (1, total_vars, 1); squeeze it to (total_vars,)
            F_val = F_val.squeeze()
            return 0.5 * torch.sum(F_val ** 2)
        
        # Flatten y for autograd.functional functions.
        y_flat = y.view(-1)
        
        # Compute the Hessian of phi_func with respect to y.
        H = torch.autograd.functional.hessian(phi_func, y_flat)
        
        # Now define a function that maps y to F (flattened).
        def F_func(y_flat):
            y_in = y_flat.view(total_vars, 1)
            x    = y_in[:num_var].unsqueeze(0)
            if num_ineq > 0:
                eta  = y_in[num_var:num_var+num_ineq].unsqueeze(0)
                s    = y_in[num_var+num_ineq:num_var+2*num_ineq].unsqueeze(0)
            else:
                eta = torch.empty((1, 0, 1), device=y.device)
                s   = torch.empty((1, 0, 1), device=y.device)
            if num_eq > 0:
                lamb = y_in[num_var+2*num_ineq:num_var+2*num_ineq+num_eq].unsqueeze(0)
            else:
                lamb = torch.empty((1, 0, 1), device=y.device)
            F_val, _ = self.calculate_F(x, eta, s, lamb, None, None, sigma, **kwargs)
            return F_val.squeeze()  # shape: (total_vars,)
        
        # Compute the Jacobian of F with respect to y.
        J = torch.autograd.functional.jacobian(F_func, y_flat)
        
        # Evaluate F at the current y.
        F_current = F_func(y_flat)
        
        return H, J, F_current


    def sub_objective(self, y, J, F):
        """
        y: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        J: [batch_size, num_var+num_ineq+num_ineq+num_eq, num_var+num_ineq+num_ineq+num_eq]
        F: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        1/2||J@y-F||_2^2 = 1/2(y^T@J^T@y)-y^TJ^TF+1/2(F^TF)
        """
        obj0 = 0.5 * torch.bmm(torch.bmm(y.permute(0, 2, 1), J.permute(0, 2, 1)), torch.bmm(J, y))
        obj1 = torch.bmm(torch.bmm(y.permute(0, 2, 1), J.permute(0, 2, 1)), F)
        obj2 = 0.5 * (torch.bmm(J.permute(0, 2, 1), F))
        return obj0 + obj1 + obj2

    def sub_smooth_grad(self, y, J, F):
        """
        y: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        return J^T@J@y+J^T@F
        """
        grad = torch.bmm(torch.bmm(J.permute(0, 2, 1), J), y) + torch.bmm(J.permute(0, 2, 1), F)
        return grad

    def opt_solve(self, solver_type='iopopt', tol=1e-4, initial_y = None, init_mu=None, init_g=None, init_zl=None, init_zu=None):
        if solver_type == 'ipopt':
            Q, p = self.Q.detach().cpu().numpy(), self.p.detach().cpu().numpy()
            if self.num_ineq != 0:
                Q_ineq, G, c = self.Q_ineq.detach().cpu().numpy(), self.G.detach().cpu().numpy(), self.c.detach().cpu().numpy()
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
            objs  = []
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
                cl = np.hstack(cls)
                cu = np.hstack(cus)

                if (self.num_ineq != 0) and (self.num_eq != 0):
                    Q_ineq, G0, A0 = Q_ineq, G[i], A[i]
                elif (self.num_ineq != 0) and (self.num_eq == 0):
                    Q_ineq, G0, A0 = Q_ineq, G[i], np.array([0.0])
                elif (self.num_ineq == 0) and (self.num_eq != 0):
                    Q_ineq, G0, A0 = np.array([0.0]), np.array([0.0]), A[i]

                nlp = convex_ipopt(
                    Q[i],
                    p[i].squeeze(-1),
                    Q_ineq,
                    G0,
                    A0,
                    n=len(y0),
                    m=len(cl),
                    # problem_obj=prob_obj,
                    lb=lb[i],
                    ub=ub[i],
                    cl=cl,
                    cu=cu
                )

                nlp.add_option('tol', tol)
                nlp.tol_resid = tol
                #nlp.add_option('nlp_scaling_method', 'none')
                #nlp.add_option('jacobian_scaling',   'none')
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
                obj_val = nlp.objective(y)
                objs.append(obj_val)

                end_time = time.time()
                Y.append(y)
                iters.append(len(nlp.objectives))
                total_time += (end_time - start_time)

            sols = np.array(Y)
            parallel_time = total_time / Q.shape[0]
        else:
            raise NotImplementedError
        
        obj_vals = np.array(objs)

        return sols, total_time, parallel_time, np.array(iters).mean(), nlp.func_evals, obj_vals
    
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
    def __init__(self, Q, p, Q_ineq, G, A, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residuals = []
        self.Q = Q
        self.p = p
        self.Q_ineq = Q_ineq
        self.G = G
        self.A = A
        self.func_evals = 0
        self.grad_evals = 0
        self.cons_evals = 0
        self.jac_evals  = 0
        if ((self.G == 0.0).all()) and (len(self.G)==1):
            self.num_ineq = 0
        else:
            self.num_ineq = self.G.shape[0]
        if ((self.A == 0.0).all()) and (len(self.A)==1):
            self.num_eq = 0
        else:
            self.num_eq = self.A.shape[0]

        self.objectives = []
        self.mus = []
        self.tril_indices = np.tril_indices(Q.shape[0])

    def objective(self, y):
        self.func_evals += 1
        return 0.5 * (y @ self.Q @ y) + self.p@y

    def gradient(self, y):
        self.grad_evals += 1
        return self.Q@y + self.p

    def constraints(self, y):
        self.cons_evals += 1
        const_values = []
        if self.num_ineq != 0:
            if (self.Q_ineq == 0).all():
                const_values.append(self.G @ y)
            else:
                ineq_const = []
                for i in range(self.num_ineq):
                    ineq_const.append(y.T@self.Q_ineq[i]@y+self.G[i,:].T@y)
                const_values.append(np.array(ineq_const))
        if self.num_eq != 0:
            const_values.append(self.A @ y)
        return np.hstack(const_values)


    def jacobian(self, y):
        self.jac_evals += 1
        const_jacob = []
        if self.num_ineq != 0:
            if (self.Q_ineq == 0).all():
                const_jacob.append(self.G.flatten())
            else:
                ineq_grad = []
                for i in range(self.num_ineq):
                    ineq_grad.append((self.Q_ineq[i,:,:].T+self.Q_ineq[i,:,:])@y+self.G[i,:])
                const_jacob.append(np.concatenate(ineq_grad, axis=-1).T.flatten())
        if self.num_eq != 0:
            const_jacob.append(self.A.flatten())
        return np.concatenate(const_jacob)

    def intermediate(self, alg_mod, iter_count, obj_value,
            inf_pr, inf_du, mu, d_norm, regularization_size,
            alpha_du, alpha_pr, ls_trials):
        self.objectives.append(obj_value)
        self.mus.append(mu)

        it = self.get_current_iterate(scaled=True)
        x        = it["x"]            # shape (n,)
        mult_x_L = it["mult_x_L"]     # λ_L for x ≥ x_L
        mult_x_U = it["mult_x_U"]     # λ_U for x ≤ x_U
        g_vals   = it["g"]            # g(x)
        mult_g   = it["mult_g"]       # Lagrange multipliers for g(x)

        # 2) grab the **violation** vectors (gradient of Lagrangian, constraint violations, complementarity, etc.)
        #    again un‑scaled so they match your raw KKT residual :contentReference[oaicite:1]{index=1}
        v = self.get_current_violations(scaled=True)
        grad_lag    = v["grad_lag_x"]        # ∇ₓL
        g_violation = v["g_violation"]      # g(x) – g_U  and/or g_L – g(x)
        compl_g     = v["compl_g"]          # dual·primal complementarity for g
        xL_viol     = v["x_L_violation"]    # x_L – x   (when x < x_L)
        xU_viol     = v["x_U_violation"]    # x – x_U   (when x > x_U)
        compl_xL    = v["compl_x_L"]        # λ_L·(x_L – x)
        compl_xU    = v["compl_x_U"]        # λ_U·(x – x_U)

        # 3) assemble your full F exactly as in calculate_F:
        #    [stationarity; primal infeas; comp_ineq; bound infeas; comp_bounds]
        F = np.concatenate([
            grad_lag,           # F1
            g_violation,        # F2
            compl_g,            # F3
            xL_viol,            # F5
            xU_viol,            # F6
            compl_xL,           # F5 comp
            compl_xU,           # F6 comp
        ])

        raw_resid = np.linalg.norm(F, 2)
        self.residuals.append(raw_resid)
        print(f"IPOPT raw ‖F‖₂ = {raw_resid:.4e}")

        if raw_resid < getattr(self, "tol_resid", np.inf):
            print(f"Stopping IPOPT at iter {iter_count}, raw‖F‖₂={raw_resid:.4e}")
            return False
        

        # resid = np.sqrt(inf_pr**2 + inf_du**2)
        # self.residuals.append(resid)

        # # 2) print it every iteration
        # print(f"IPOPT iter {iter_count:3d} | Residual ‖F‖₂ = {resid:.4e}")
        #resid = np.sqrt(inf_pr**2 + inf_du**2)
        #self.residuals.append(resid)

    


    

    

