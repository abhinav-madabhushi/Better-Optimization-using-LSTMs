#import ctypes
#ctypes.windll.kernel32.SetDllDirectoryW(r"C:\Users\krish\Ipopt-3.14.17-win64-msvs2022-md\Ipopt-3.14.17-win64-msvs2022-md\bin")
import os
import sys
import numpy as np
import configargparse
import time
import torch
import torch.optim as optim
import scipy.io as sio
import subprocess
import json

import torch
from torch.utils.data import Dataset  

from models.LSTM import LSTM
from problems.QP import QP
from problems.Convex_QCQP import Convex_QCQP
from problems.Nonconvex_Program import Nonconvex_Program
#from problems.CUTEST import CUTEST as CUTESTDataset
from utils import EarlyStopping, calculate_step
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pad_array(arr, target_shape):
    """
    Pads the tensor `arr` with zeros to match `target_shape`.
    Assumes arr is a torch.Tensor.
    """
    # If the array has fewer dimensions than the target, add extra dimensions
    if arr.dim() < len(target_shape):
        for _ in range(len(target_shape) - arr.dim()):
            arr = arr.unsqueeze(-1)
    padded = torch.zeros(target_shape, dtype=arr.dtype, device=arr.device)
    slices = tuple(slice(0, s) for s in arr.shape)
    padded[slices] = arr
    return padded

def custom_collate_fn(batch, target_n=100, target_eq=50, target_ineq=50):
    Q_list, p_list, A_list, b_list, G_list, c_list, lb_list, ub_list, idx_list = [], [], [], [], [], [], [], [], []
    for item in batch:
        # Instead of using the actual size, pad everything to the fixed target dimensions.
        Q_list.append(pad_array(item["Q"], (target_n, target_n)))
        p_list.append(pad_array(item["p"], (target_n, 1)))
        if item["A"] is not None:
            A_list.append(pad_array(item["A"], (target_eq, target_n)))
            b_list.append(pad_array(item["b"], (target_eq, 1)))
        else:
            A_list.append(torch.zeros((target_eq, target_n), dtype=item["Q"].dtype, device=item["Q"].device))
            b_list.append(torch.zeros((target_eq, 1), dtype=item["Q"].dtype, device=item["Q"].device))
        if item["G"] is not None:
            G_list.append(pad_array(item["G"], (target_ineq, target_n)))
            c_list.append(pad_array(item["c"], (target_ineq, 1)))
        else:
            G_list.append(torch.zeros((target_ineq, target_n), dtype=item["Q"].dtype, device=item["Q"].device))
            c_list.append(torch.zeros((target_ineq, 1), dtype=item["Q"].dtype, device=item["Q"].device))
        lb_list.append(pad_array(item["lb"], (target_n, 1)))
        ub_list.append(pad_array(item["ub"], (target_n, 1)))
        idx_list.append(item["idx"])
    
    batch_dict = {
        "Q": torch.stack(Q_list, dim=0),
        "p": torch.stack(p_list, dim=0),
        "A": torch.stack(A_list, dim=0),
        "b": torch.stack(b_list, dim=0),
        "G": torch.stack(G_list, dim=0),
        "c": torch.stack(c_list, dim=0),
        "lb": torch.stack(lb_list, dim=0),
        "ub": torch.stack(ub_list, dim=0),
        "max_n": target_n,
        "max_eq": target_eq,
        "max_ineq": target_ineq,
        "idx": torch.tensor(idx_list, device=DEVICE)
    }
    return batch_dict




# ---------------------------
# Parse command-line arguments
# ---------------------------
parser = configargparse.ArgumentParser(description='train')
parser.add_argument('-c', '--config', is_config_file=True, type=str)

# Optimizee settings
parser.add_argument('--mat_name', type=str, help='Imported mat file name.')
parser.add_argument('--num_var', type=int, help='Number of decision vars.')
parser.add_argument('--num_eq', type=int, help='Number of equality constraints.')
parser.add_argument('--num_ineq', type=int, help='Number of inequality constraints.')
parser.add_argument('--prob_type', type=str, help='Problem type.')

# CUTEST-specific file arguments (for prob_type CUTEST)
parser.add_argument('--cutest_train_file', type=str, help='TXT file listing CUTEST training problems')
parser.add_argument('--cutest_val_file', type=str, help='TXT file listing CUTEST validation problems')
parser.add_argument('--cutest_test_file', type=str, help='TXT file listing CUTEST testing problems')

# Model settings
parser.add_argument('--eq_tol', type=float, help='equality tolerance for model saving.')
parser.add_argument('--ineq_tol', type=float, help='inequality tolerance for model saving.')
parser.add_argument('--input_dim', type=int, default=2, help='Input feature dimensions of deep learning optimizer.')
parser.add_argument('--hidden_dim', type=int, default=128, help='The hidden dimensions of deep learning optimizer.')
parser.add_argument('--use_line_search', action='store_true', help='Using line search.')
parser.add_argument('--model_name', type=str, help='The deep learning optimizer name.')
parser.add_argument('--precondi', action='store_true', help='Preconditioning.')
parser.add_argument('--sigma', type=float, help='The coefficient of mu.')
parser.add_argument('--tau', type=float, help='The value of frac-and-boundary.')

# Training settings
parser.add_argument('--batch_size', type=int, help='training batch size.')
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--data_size', type=int, help='The number of all instances.')
parser.add_argument('--lr', type=float, help='Learning rate.')
parser.add_argument('--num_epoch', type=int, help='The number of training epochs.')
parser.add_argument('--inner_T', type=int, default=1, help='The iterations of deep learning optimizer.')
parser.add_argument('--outer_T', type=int, help='The iterations of IPM.')
parser.add_argument('--patience', type=int, default=100, help='The patience of early stopping.')
parser.add_argument('--save_dir', type=str, default='./results/', help='Save path for the best model.')
parser.add_argument('--save_sol', action='store_true', help='Save the results.')
parser.add_argument('--seed', type=int, default=17, help='random seed.')
parser.add_argument('--test', action='store_true', help='Run in test mode.')
parser.add_argument('--test_solver', type=str, choices=['osqp', 'ipopt'], help='The solver type on the test set.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay rate.')

args, _ = parser.parse_known_args()
if args.device is None:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device(args.device)
args.device = DEVICE

# ---------------------------
# Model instantiation
# ---------------------------
if (args.model_name is None) or (args.model_name == 'LSTM'):
    model = LSTM(args.input_dim, args.hidden_dim, args.inner_T, DEVICE)
else:
    raise ValueError(f"Unsupported model_name {args.model_name}")
model = model.to(DEVICE)

# ---------------------------
# Optimizee / Dataset selection
# ---------------------------
if args.prob_type == 'Nonconvex_QP':
    optimizee = QP
    file_path = os.path.join('datasets', 'qp', "{}.mat".format(args.mat_name))
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                               args.mat_name,
                                                                               args.outer_T,
                                                                               args.inner_T))
elif args.prob_type == 'QP':
    optimizee = QP
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                                                args.num_var,
                                                                                                args.num_ineq,
                                                                                                args.num_eq,
                                                                                                args.outer_T,
                                                                                                args.inner_T))
    #save_path = os.path.join(args.save_dir, model.name(), 'params', 'portfolio_QP_box_s10_t2_eq3.pt')
    
elif args.prob_type == 'CUTEST':
    train_data = CUTESTDataset(
        learning_type='train',
        file_path=args.cutest_train_file,
        device=args.device,
        num_var=args.num_var
    )
    val_data = CUTESTDataset(
        learning_type='val',
        file_path=args.cutest_val_file,
        device=args.device,
        num_var=args.num_var
    )
    save_path = os.path.join(
        args.save_dir, model.name(), 'params',
        '{}_{}_{}_{}.pth'.format(args.prob_type, args.num_var, args.num_ineq, args.inner_T)
    )
elif args.prob_type == 'QP_RHS':
    optimizee = QP
    mat_name = args.mat_name
    file_path = args.file_path if hasattr(args, 'file_path') and args.file_path is not None else os.path.join('datasets', 'qp', "{}".format(mat_name))
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                                                   args.num_var,
                                                                                                   args.num_ineq,
                                                                                                   args.num_eq,
                                                                                                   args.outer_T,
                                                                                                   args.inner_T))
elif args.prob_type == 'Nonconvex_Program':
    optimizee = Nonconvex_Program
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                                                   args.num_var,
                                                                                                   args.num_ineq,
                                                                                                   args.num_eq,
                                                                                                   args.outer_T,
                                                                                                   args.inner_T))
elif args.prob_type == 'Nonconvex_Program_RHS':
    optimizee = Nonconvex_Program
    mat_name = "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(args.num_var, args.num_ineq, args.num_eq, args.data_size)
    file_path = os.path.join('datasets', 'nonconvex_program', "{}".format(mat_name))
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                                                   args.num_var,
                                                                                                   args.num_ineq,
                                                                                                   args.num_eq,
                                                                                                   args.outer_T,
                                                                                                   args.inner_T))
elif args.prob_type == 'Convex_QCQP_RHS':
    optimizee = Convex_QCQP
    mat_name = "random_convex_qcqp_dataset_var{}_ineq{}_eq{}_ex{}".format(args.num_var,
                                                                         args.num_ineq,
                                                                         args.num_eq,
                                                                         args.data_size)
    file_path = os.path.join('datasets', 'convex_qcqp', "{}.mat".format(mat_name))
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                                                   args.num_var,
                                                                                                   args.num_ineq,
                                                                                                   args.num_eq,
                                                                                                   args.outer_T,
                                                                                                   args.inner_T))
elif args.prob_type == 'Convex_QCQP':
    optimizee = Convex_QCQP
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                                                   args.num_var,
                                                                                                   args.num_ineq,
                                                                                                   args.num_eq,
                                                                                                   args.outer_T,
                                                                                                   args.inner_T))

# ---------------------------
# Training/Validation Phase
# ---------------------------
if not args.test:
    stopper = EarlyStopping(save_path, patience=args.patience)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.prob_type in ['QP', 'Nonconvex_QP', 'Convex_QCQP']:
        train_data = optimizee(prob_type=args.prob_type, learning_type='train', num_var=args.num_var,
                            num_ineq=args.num_ineq, num_eq=args.num_eq, data_size=args.data_size, device=DEVICE)
        val_data = optimizee(prob_type=args.prob_type, learning_type='val', num_var=args.num_var,
                            num_ineq=args.num_ineq, num_eq=args.num_eq, data_size=args.data_size, device=DEVICE)
    elif args.prob_type in ['QP_RHS', 'Nonconvex_Program_RHS', 'Convex_QCQP_RHS']:
        train_data = optimizee(prob_type=args.prob_type, learning_type='train', file_path=file_path,
                            num_var=args.num_var, num_ineq=args.num_ineq, num_eq=args.num_eq, data_size=args.data_size, device=DEVICE)
        val_data = optimizee(prob_type=args.prob_type, learning_type='val', file_path=file_path,
                            num_var=args.num_var, num_ineq=args.num_ineq, num_eq=args.num_eq, data_size=args.data_size, device=DEVICE)

    elif args.prob_type == 'CUTEST':
        pass
    else:
        train_data = optimizee(prob_type=args.prob_type, learning_type='train', file_path=file_path, device=DEVICE)
        val_data = optimizee(prob_type=args.prob_type, learning_type='val', file_path=file_path, device=DEVICE)
    
    print(f"Loaded Data - num_var: {train_data.num_var}, num_eq: {train_data.num_eq}, num_ineq: {train_data.num_ineq}")
    print('The number of variables: {}.'.format(train_data.num_var))
    if train_data.num_ineq != 0:
        print('The number of inequalities: {}.'.format(train_data.num_ineq))
    if train_data.num_eq != 0:
        print('The number of equalities: {}.'.format(train_data.num_eq))
    if train_data.num_lb != 0:
        print('The number of lower bounds: {}.'.format(train_data.num_lb))
    if train_data.num_ub != 0:
        print('The number of upper bounds: {}.'.format(train_data.num_ub))
    
    if args.prob_type == 'CUTEST':
        val_batch_size = min(1000, len(val_data))
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False, collate_fn=custom_collate_fn)
    else:
        train_parameters = [train_data.Q, train_data.p]
        if train_data.num_ineq != 0:
            train_parameters.append(train_data.G)
            train_parameters.append(train_data.c)
        if train_data.num_eq != 0:
            train_parameters.append(train_data.A)
            train_parameters.append(train_data.b)
        if train_data.num_lb != 0:
            train_parameters.append(train_data.lb)
        if train_data.num_ub != 0:
            train_parameters.append(train_data.ub)
        train_dataset = TensorDataset(*train_parameters)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        val_parameters = [val_data.Q, val_data.p]
        if val_data.num_ineq != 0:
            val_parameters.append(val_data.G)
            val_parameters.append(val_data.c)
        if val_data.num_eq != 0:
            val_parameters.append(val_data.A)
            val_parameters.append(val_data.b)
        if val_data.num_lb != 0:
            val_parameters.append(val_data.lb)
        if val_data.num_ub != 0:
            val_parameters.append(val_data.ub)
        val_dataset = TensorDataset(*val_parameters)
        val_batch_size = min(1000, len(val_dataset))
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    
    for epoch in range(args.num_epoch):
        model.train()
        train_start_time = time.time()
        for batch in train_loader:
            if args.prob_type == 'CUTEST':
                train_Q = batch["Q"]
                train_p = batch["p"]
                train_A = batch["A"]
                train_b = batch["b"]
                train_G = batch["G"]
                train_c = batch["c"]
                train_lb = batch["lb"]
                train_ub = batch["ub"]
            else:
                train_Q, train_p, *remaining_parameters0 = batch
                if train_data.num_ineq != 0:
                    train_G, train_c, *remaining_parameters1 = remaining_parameters0
                    if train_data.num_eq != 0:
                        train_A, train_b, *remaining_parameters3 = remaining_parameters1
                        if train_data.num_lb != 0:
                            train_lb, *remaining_parameters5 = remaining_parameters3
                            if train_data.num_ub != 0:
                                train_ub = remaining_parameters5[0]
                            else:
                                train_ub = None
                        else:
                            train_lb = None
                            if train_data.num_ub != 0:
                                train_ub = remaining_parameters3[0]
                            else:
                                train_ub = None
                    else:
                        train_A = None
                        train_b = None
                        if train_data.num_lb != 0:
                            train_lb, *remaining_parameters7 = remaining_parameters1
                            if train_data.num_ub != 0:
                                train_ub = remaining_parameters7[0]
                            else:
                                train_ub = None
                        else:
                            train_lb = None
                            if train_data.num_ub != 0:
                                train_ub = remaining_parameters1[0]
                            else:
                                train_ub = None
                else:
                    train_G = None
                    train_c = None
                    if train_data.num_eq != 0:
                        train_A, train_b, *remaining_parameters2 = remaining_parameters0
                        if train_data.num_lb != 0:
                            train_lb, *remaining_parameters4 = remaining_parameters2
                            if train_data.num_ub != 0:
                                train_ub = remaining_parameters4[0]
                            else:
                                train_ub = None
                        else:
                            train_lb = None
                            if train_data.num_ub != 0:
                                train_ub = remaining_parameters2[0]
                            else:
                                train_ub = None
                    else:
                        train_A = None
                        train_b = None
                        if train_data.num_lb != 0:
                            train_lb, *remaining_parameters6 = remaining_parameters0
                            if train_data.num_ub != 0:
                                train_ub = remaining_parameters6[0]
                            else:
                                train_ub = None
                        else:
                            train_lb = None
                            if train_data.num_ub != 0:
                                train_ub = remaining_parameters0[0]
                            else:
                                train_ub = None

            if (train_data.num_lb != 0) and (train_data.num_ub != 0):
                train_x = (train_lb+train_ub)/2
            elif (train_data.num_lb != 0) and (train_data.num_ub == 0):
                train_x = train_lb + torch.ones(size=train_lb.shape, device=args.device)
            elif (train_data.num_lb == 0) and (train_data.num_ub != 0):
                train_x = train_ub - torch.ones(size=train_ub.shape, device=args.device)
            else:
                train_x = torch.zeros((train_Q.shape[0], train_data.num_var, 1), device=args.device)

            if train_data.num_ineq != 0:
                train_eta = torch.ones((train_Q.shape[0], train_data.num_ineq, 1), device=args.device)
                train_s = torch.ones((train_Q.shape[0], train_data.num_ineq, 1), device=args.device)
            else:
                train_eta = None
                train_s = None
            if train_data.num_eq != 0:
                train_lamb = torch.zeros((train_Q.shape[0], train_data.num_eq, 1), device=args.device)
            else:
                train_lamb = None
            if train_data.num_lb != 0:
                train_zl = torch.ones((train_Q.shape[0], train_data.num_lb, 1), device=args.device)
            else:
                train_zl = None
            if train_data.num_ub != 0:
                train_zu = torch.ones((train_Q.shape[0], train_data.num_ub, 1), device=args.device)
            else:
                train_zu = None

            for t_out in range(args.outer_T):
                train_J, train_F, train_mu = train_data.cal_kkt(train_x, train_eta, train_s, train_lamb, train_zl, train_zu, args.sigma,
                                                                Q=train_Q, p=train_p, G=train_G, c=train_c, A=train_A, b=train_b,
                                                                lb=train_lb, ub=train_ub)
                train_res_norm = train_F.norm().item()
                try:
                    val_res_norm = val_F.norm().item()
                except:
                    val_res_norm = 0.0

                init_y = torch.zeros((train_Q.shape[0], train_data.num_var+2*train_data.num_ineq+train_data.num_eq+train_data.num_lb+train_data.num_ub, 1), device=args.device)

                if args.precondi:
                    train_D_values, train_D_id = (torch.bmm(train_J.permute(0, 2, 1), train_J)).max(-1)
                    train_D_inverse = torch.diag_embed(torch.sqrt(1 / train_D_values))
                    train_J_0 = torch.bmm(train_J, train_D_inverse)
                else:
                    train_J_0 = train_J

                train_y, train_loss, _ = model(train_data, init_y, train_J_0, train_F)
                optimizer.zero_grad()
                train_loss.backward(retain_graph=True)
                optimizer.step()
                if args.precondi:
                    train_y = torch.bmm(train_D_inverse, train_y)

                delta_x = train_y[:, :train_data.num_var, :]
                delta_eta = train_y[:, train_data.num_var:train_data.num_var + train_data.num_ineq, :]
                delta_lamb = train_y[:, train_data.num_var + train_data.num_ineq:train_data.num_var + train_data.num_ineq + train_data.num_eq, :]
                delta_s = train_y[:, train_data.num_var + train_data.num_ineq + train_data.num_eq:train_data.num_var + 2*train_data.num_ineq+train_data.num_eq, :]
                delta_zl = train_y[:, train_data.num_var + 2*train_data.num_ineq+train_data.num_eq:train_data.num_var + 2*train_data.num_ineq+train_data.num_eq+train_data.num_lb, :]
                delta_zu = train_y[:, train_data.num_var+2*train_data.num_ineq+train_data.num_eq+train_data.num_lb:, :]
                alpha_x, alpha_eta, alpha_s, alpha_zl, alpha_zu = calculate_step(train_data, train_x, delta_x, train_eta, delta_eta, train_s,
                                                                                delta_s, train_zl, delta_zl, train_zu, delta_zu, train_lb,
                                                                                train_ub, args.tau, args.use_line_search, device=DEVICE)
                if (train_data.num_lb != 0) or (train_data.num_ub != 0):
                    train_x = (train_x+alpha_x*delta_x).detach()
                else:
                    if train_data.num_ineq != 0:
                        train_x = (train_x + alpha_s * delta_x).detach()
                    else:
                        train_x = (train_x + delta_x).detach()
                if train_data.num_ineq != 0:
                    train_eta = (train_eta+alpha_eta*delta_eta).detach()
                    train_s = (train_s+alpha_s*delta_s).detach()
                if train_data.num_eq != 0:
                    if (train_data.num_lb != 0) or (train_data.num_ub != 0):
                        train_lamb = (train_lamb+alpha_x*delta_lamb).detach()
                    else:
                        if train_data.num_ineq != 0:
                            train_lamb = (train_lamb + alpha_s * delta_lamb).detach()
                        else:
                            train_lamb = (train_lamb + delta_lamb).detach()
                if train_data.num_lb != 0:
                    train_zl = (train_zl+alpha_zl*delta_zl).detach()
                if train_data.num_ub != 0:
                    train_zu = (train_zu+alpha_zu*delta_zu).detach()

        train_end_time = time.time()
        train_obj = train_data.obj_fn(train_x, Q=train_Q, p=train_p).mean()
        if train_data.num_ineq != 0:
            train_ineq_vio_max = train_data.ineq_dist(train_x, G=train_G, c=train_c).max(dim=1).values.mean()
            train_ineq_vio_mean = train_data.ineq_dist(train_x, G=train_G, c=train_c).mean()
        if train_data.num_eq != 0:
            train_eq_vio_max = train_data.eq_dist(train_x, A=train_A, b=train_b).max(dim=1).values.mean()
            train_eq_vio_mean = train_data.eq_dist(train_x, A=train_A, b=train_b).mean()
        if train_data.num_lb != 0:
            train_lb_vio_max = train_data.lower_bound_dist(train_x, lb=train_lb).max(dim=1).values.mean()
            train_lb_vio_mean = train_data.lower_bound_dist(train_x, lb=train_lb).mean()
        if train_data.num_ub != 0:
            train_ub_vio_max = train_data.upper_bound_dist(train_x, ub=train_ub).max(dim=1).values.mean()
            train_ub_vio_mean = train_data.upper_bound_dist(train_x, ub=train_ub).mean()

        model.eval()
        with torch.no_grad():
            if (val_data.num_lb != 0) and (val_data.num_ub != 0):
                val_x = (val_data.lb + val_data.ub) / 2
            elif (val_data.num_lb != 0) and (val_data.num_ub == 0):
                val_x = val_data.lb + torch.ones(size=val_data.lb.shape, device=args.device)
            elif (val_data.num_lb == 0) and (val_data.num_ub != 0):
                val_x = val_data.ub - torch.ones(size=val_data.ub.shape, device=args.device)
            else:
                val_x = torch.zeros((val_data.Q.shape[0], val_data.num_var, 1), device=args.device)
            if val_data.num_ineq != 0:
                val_eta = torch.ones((val_data.Q.shape[0], val_data.num_ineq, 1), device=args.device)
                val_s = torch.ones((val_data.Q.shape[0], val_data.num_ineq, 1), device=args.device)
            else:
                val_eta = None
                val_s = None
            if val_data.num_eq != 0:
                val_lamb = torch.zeros((val_data.Q.shape[0], val_data.num_eq, 1), device=args.device)
            else:
                val_lamb = None
            if val_data.num_lb != 0:
                val_zl = torch.ones((val_data.Q.shape[0], val_data.num_lb, 1), device=args.device)
            else:
                val_zl = None
            if val_data.num_ub != 0:
                val_zu = torch.ones((val_data.Q.shape[0], val_data.num_ub, 1), device=args.device)
            else:
                val_zu = None
            val_start_time = time.time()
            for t_out in range(args.outer_T):
                val_J, val_F, val_mu = val_data.cal_kkt(val_x, val_eta, val_s, val_lamb, val_zl, val_zu, args.sigma)
                init_y = torch.zeros((val_data.val_size, val_data.num_var+2*val_data.num_ineq+val_data.num_eq+val_data.num_lb+val_data.num_ub, 1), device=args.device)
                if args.precondi:
                    val_D_values, D_id = (torch.bmm(val_J.permute(0, 2, 1), val_J)).max(-1)
                    val_D_inverse = torch.diag_embed(torch.sqrt(1 / val_D_values))
                    val_J_0 = torch.bmm(val_J, val_D_inverse)
                else:
                    val_J_0 = val_J
                val_y, val_loss, _ = model(val_data, init_y, val_J_0, val_F)
                if args.precondi:
                    val_y = torch.bmm(val_D_inverse, val_y)
                delta_x = val_y[:, :val_data.num_var, :]
                delta_eta = val_y[:, val_data.num_var:val_data.num_var + val_data.num_ineq, :]
                delta_lamb = val_y[:, val_data.num_var + val_data.num_ineq:val_data.num_var + val_data.num_ineq + val_data.num_eq, :]
                delta_s = val_y[:, val_data.num_var + val_data.num_ineq + val_data.num_eq:val_data.num_var + 2 * val_data.num_ineq + val_data.num_eq, :]
                delta_zl = val_y[:, val_data.num_var + 2 * val_data.num_ineq + val_data.num_eq:val_data.num_var + 2 * val_data.num_ineq + val_data.num_eq + val_data.num_lb, :]
                delta_zu = val_y[:, val_data.num_var + 2 * val_data.num_ineq + val_data.num_eq + val_data.num_lb:, :]
                alpha_x, alpha_eta, alpha_s, alpha_zl, alpha_zu = calculate_step(val_data, val_x, delta_x, val_eta, delta_eta,
                                                                                val_s, delta_s, val_zl, delta_zl, val_zu,
                                                                                delta_zu, val_data.lb, val_data.ub, args.tau,
                                                                                args.use_line_search, device=DEVICE)
                if (val_data.num_lb != 0) or (val_data.num_ub != 0):
                    val_x = (val_x + alpha_x * delta_x).detach()
                else:
                    if val_data.num_ineq != 0:
                        val_x = (val_x + alpha_s * delta_x).detach()
                    else:
                        val_x = (val_x + delta_x).detach()
                if val_data.num_ineq != 0:
                    val_eta = (val_eta + alpha_eta * delta_eta).detach()
                    val_s = (val_s + alpha_s * delta_s).detach()
                if val_data.num_eq != 0:
                    if (val_data.num_lb != 0) or (val_data.num_ub != 0):
                        val_lamb = (val_lamb+alpha_x*delta_lamb).detach()
                    else:
                        if val_data.num_ineq != 0:
                            val_lamb = (val_lamb + alpha_s * delta_lamb).detach()
                        else:
                            val_lamb = (val_lamb + delta_lamb).detach()
                if val_data.num_lb != 0:
                    val_zl = (val_zl + alpha_zl * delta_zl).detach()
                if val_data.num_ub != 0:
                    val_zu = (val_zu + alpha_zu * delta_zu).detach()
            val_end_time = time.time()
            val_vios = []
            val_obj = val_data.obj_fn(val_x).mean()
            if val_data.num_ineq != 0:
                val_ineq_vio_max = val_data.ineq_dist(val_x).max(dim=1).values.mean()
                val_ineq_vio_mean = val_data.ineq_dist(val_x).mean()
                val_vios.append(val_ineq_vio_max.data.item())
            if val_data.num_eq != 0:
                val_eq_vio_max = val_data.eq_dist(val_x).max(dim=1).values.mean()
                val_eq_vio_mean = val_data.eq_dist(val_x).mean()
                val_vios.append(val_eq_vio_max.data.item())
            if val_data.num_lb != 0:
                val_lb_vio_max = val_data.lower_bound_dist(val_x).max(dim=1).values.mean()
                val_lb_vio_mean = val_data.lower_bound_dist(val_x).mean()
                val_vios.append(val_lb_vio_max.data.item())
            if val_data.num_ub != 0:
                val_ub_vio_max = val_data.upper_bound_dist(val_x).max(dim=1).values.mean()
                val_ub_vio_mean = val_data.upper_bound_dist(val_x).mean()
                val_vios.append(val_ub_vio_max.data.item())
        early_stop = stopper.step(val_obj.data.item(), model, args.eq_tol, *val_vios)
        print("Epoch : {} | Train_Obj : {:.3f} | Val_Obj : {:.3f} | Train_Res_Norm: {:.3f} | Val_Res_Norm: {:.3f} | Train_Time : {:.3f} | Val_Time : {:.3f} |".format(epoch, train_obj, val_obj, train_res_norm, val_res_norm, train_end_time - train_start_time, val_end_time - val_start_time))
        if val_data.num_ineq != 0:
            print("Epoch : {} | Train_Max_Ineq : {:.3f} | Train_Mean_Ineq : {:.3f} | Val_Max_Ineq : {:.3f} | Val_Mean_Ineq : {:.3f} |".format(epoch, train_ineq_vio_max, train_ineq_vio_mean, val_ineq_vio_max, val_ineq_vio_mean))
        if val_data.num_eq != 0:
            print("Epoch : {} | Train_Max_Eq : {:.3f} | Train_Mean_Eq : {:.3f} | Val_Max_Eq : {:.3f} | Val_Mean_Eq : {:.3f} |".format(epoch, train_eq_vio_max, train_eq_vio_mean, val_eq_vio_max, val_eq_vio_mean))
        if val_data.num_lb != 0:
            print("Epoch : {} | Train_Max_Lb : {:.3f} | Train_Mean_Lb : {:.3f} | Val_Max_Lb : {:.3f} | Val_Mean_Lb : {:.3f} |".format(epoch, train_lb_vio_max, train_lb_vio_mean, val_lb_vio_max, val_lb_vio_mean))
        if val_data.num_ub != 0:
            print("Epoch : {} | Train_Max_Ub : {:.3f} | Train_Mean_Ub : {:.3f} | Val_Max_Ub : {:.3f} | Val_Mean_Ub : {:.3f} |".format(epoch, train_ub_vio_max, train_ub_vio_mean, val_ub_vio_max, val_ub_vio_mean))
        # convergence diagnostics using basic residuals
        thresh = 1e-2
        train_P = train_data.primal_feasibility_basic(train_x, s=train_s, lb=train_lb, ub=train_ub, A=train_A, b=train_b, G=train_G, c=train_c)
        train_D = train_data.dual_feasibility_basic(train_x, eta=train_eta, lamb=train_lamb, zl=train_zl, zu=train_zu, lb=train_lb, ub=train_ub, G=train_G, A=train_A)
        train_C = train_data.complementarity_basic(train_x, s=train_s, eta=train_eta, zl=train_zl, zu=train_zu, lb=train_lb, ub=train_ub)
        ok_train = (train_P < thresh) & (train_D < thresh) & (train_C < thresh)
        print("Train percentage converged: ", ok_train.float().mean())

        val_P = val_data.primal_feasibility_basic(val_x, s=val_s, lb=val_data.lb if val_data.num_lb != 0 else None,
                                                  ub=val_data.ub if val_data.num_ub != 0 else None,
                                                  A=val_data.A if val_data.num_eq != 0 else None,
                                                  b=val_data.b if val_data.num_eq != 0 else None,
                                                  G=val_data.G if val_data.num_ineq != 0 else None,
                                                  c=val_data.c if val_data.num_ineq != 0 else None)
        val_D = val_data.dual_feasibility_basic(val_x, eta=val_eta, lamb=val_lamb, zl=val_zl, zu=val_zu,
                                                lb=val_data.lb if val_data.num_lb != 0 else None,
                                                ub=val_data.ub if val_data.num_ub != 0 else None,
                                                G=val_data.G if val_data.num_ineq != 0 else None,
                                                A=val_data.A if val_data.num_eq != 0 else None)
        val_C = val_data.complementarity_basic(val_x, s=val_s, eta=val_eta, zl=val_zl, zu=val_zu,
                                               lb=val_data.lb if val_data.num_lb != 0 else None,
                                               ub=val_data.ub if val_data.num_ub != 0 else None)
        ok_val = (val_P < thresh) & (val_D < thresh) & (val_C < thresh)
        print("Val percentage converged: ", ok_val.float().mean())

        # primal/dual infeasibility percentages (eP/eD)
        eP_train, eD_train = train_data.primal_dual_infeasibility_basic(train_x, s=train_s, lb=train_lb, ub=train_ub,
                                                                        A=train_A, b=train_b, G=train_G, c=train_c,
                                                                        eta=train_eta, lamb=train_lamb, zl=train_zl, zu=train_zu)
        ok_train_pd = (eP_train < thresh) & (eD_train < thresh)
        print("Train percentage converged (eP/eD): ", ok_train_pd.float().mean())

        eP_val, eD_val = val_data.primal_dual_infeasibility_basic(val_x, s=val_s,
                                                                  lb=val_data.lb if val_data.num_lb != 0 else None,
                                                                  ub=val_data.ub if val_data.num_ub != 0 else None,
                                                                  A=val_data.A if val_data.num_eq != 0 else None,
                                                                  b=val_data.b if val_data.num_eq != 0 else None,
                                                                  G=val_data.G if val_data.num_ineq != 0 else None,
                                                                  c=val_data.c if val_data.num_ineq != 0 else None,
                                                                  eta=val_eta, lamb=val_lamb, zl=val_zl, zu=val_zu)
        ok_val_pd = (eP_val < thresh) & (eD_val < thresh)
        print("Val percentage converged (eP/eD): ", ok_val_pd.float().mean())
        if early_stop:
            break

elif args.test:
    if args.prob_type == 'CUTEST':
        # Load CUTEST test data and create a DataLoader using the custom collate function.
        test_data = CUTESTDataset(
            learning_type='test',
            file_path=args.cutest_test_file,
            device=args.device,
            num_var=args.num_var
        )
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
        )
        
        # Set load_path to your desired model file.
        load_path = os.path.join(
            args.save_dir,
            model.name(),
            'params',
            'Convex_QCQP_RHS_{}_{}_{}_{}_{}.pth'.format(
                args.num_var, args.num_ineq, args.num_eq, args.outer_T, args.inner_T
            )
        )
        model.load_state_dict(torch.load(load_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        # Print dataset-level diagnostics.
        print('The number of variables: {}.'.format(test_data.num_var))
        print('The number of inequalities: {}.'.format(test_data.num_ineq))
        print('The number of equalities: {}.'.format(test_data.num_eq))
        print('The number of lower bounds: {}.'.format(test_data.num_lb))
        print('The number of upper bounds: {}.'.format(test_data.num_ub))
        
        total_time = 0.0
        test_losses = []
        test_objs = []
        test_residual = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Extract padded tensors from the batch.
                test_Q  = batch["Q"]
                test_p  = batch["p"]
                test_A  = batch["A"]
                test_b  = batch["b"]
                test_G  = batch["G"]
                test_c  = batch["c"]
                test_lb = batch["lb"]
                test_ub = batch["ub"]
                
                # Set an initial guess.
                # Suppose 'batch' is the dictionary from your collate_fn, containing indices.
                # We'll create test_x for the entire batch, padded to 100 if needed.

                batch_size = batch["idx"].shape[0]
                max_n = batch["max_n"]  # e.g. 100
                init_x = torch.zeros(batch_size, max_n, 1, device=args.device)

                for i in range(batch_size):
                    # Retrieve the i-th problem
                    prob_idx = batch["idx"][i].item()
                    prob = test_data.problems[prob_idx]
                    
                    n_actual = prob.n  # e.g. 83
                    x0_np = prob.x0  # shape [n_actual]
                    
                    # Turn it into a (n_actual, 1) tensor
                    x0_torch = torch.tensor(x0_np, dtype=torch.float32, device=args.device).unsqueeze(-1)
                    
                    # Place it in init_x
                    init_x[i, :n_actual, 0] = x0_torch[:, 0]

                test_x = init_x


                            
                # Print batch dimensions for reference.
                print("Batch dimensions: n = {}, m_eq = {}, m_ineq = {}.".format(
                    batch["max_n"], batch["max_eq"], batch["max_ineq"]
                ))
                
                # Initialize dual variables.
                if test_data.num_ineq != 0:
                    test_eta = torch.ones((test_Q.shape[0], batch["max_ineq"], 1), device=args.device)
                    test_s   = torch.ones((test_Q.shape[0], batch["max_ineq"], 1), device=args.device)
                else:
                    test_eta = None
                    test_s = None
                if test_data.num_eq != 0:
                    test_lamb = torch.zeros((test_Q.shape[0], batch["max_eq"], 1), device=args.device)
                else:
                    test_lamb = None
                if test_data.num_lb != 0:
                    test_zl = torch.ones((test_Q.shape[0], batch["max_n"], 1), device=args.device)
                else:
                    test_zl = None
                if test_data.num_ub != 0:
                    test_zu = torch.ones((test_Q.shape[0], batch["max_n"], 1), device=args.device)
                else:
                    test_zu = None
                
                # Outer iterations for this batch.
                for t_out in range(args.outer_T):
                    start_time = time.time()
                    
                    # Compute the KKT system.
                    test_J, test_F, test_mu = test_data.cal_kkt(
                        test_x, test_eta, test_s, test_lamb, test_zl, test_zu, args.sigma,
                        Q=test_Q, p=test_p, A=test_A, b=test_b, G=test_G, c=test_c,
                        lb=test_lb, ub=test_ub,
                        indices=batch["idx"],
                        max_n=batch["max_n"],
                        max_eq=batch["max_eq"],
                        max_ineq=batch["max_ineq"]
                    )


                    
                    res_norm = test_F.norm().item()
                    print(f"Iter: {t_out:3d} | Residual Norm: {res_norm:8.4f}")
                    
                    # Prepare initial y.
                    init_y = torch.zeros(
                        (test_Q.shape[0],
                         test_data.num_var + 2*test_data.num_ineq + test_data.num_eq + test_data.num_lb + test_data.num_ub,
                         1),
                        device=args.device
                    )

                    

                    if args.precondi:
                        test_D_values, _ = (torch.bmm(test_J.permute(0, 2, 1), test_J)).max(-1)
                        eps = 1e-12
                        test_D_values_clamped = torch.clamp(test_D_values, min=eps)
                        test_D_inverse = torch.diag_embed(torch.sqrt(1.0 / test_D_values_clamped))
                        test_J_0 = torch.bmm(test_J, test_D_inverse)
                    else:
                        test_J_0 = test_J

                    # Run the model.
                    test_y, _, losses = model(test_data, init_y, test_J_0, test_F)
                    if args.precondi:
                        test_y = torch.bmm(test_D_inverse, test_y)
                    
                    test_losses.append(losses)


                    
                    # Extract update directions.
                    delta_x    = test_y[:, :test_data.num_var, :]
                    delta_eta  = test_y[:, test_data.num_var : test_data.num_var + test_data.num_ineq, :]
                    delta_lamb = test_y[:, test_data.num_var + test_data.num_ineq : test_data.num_var + test_data.num_ineq + test_data.num_eq, :]
                    delta_s    = test_y[:, test_data.num_var + test_data.num_ineq + test_data.num_eq : test_data.num_var + 2*test_data.num_ineq + test_data.num_eq, :]
                    delta_zl   = test_y[:, test_data.num_var + 2*test_data.num_ineq + test_data.num_eq : test_data.num_var + 2*test_data.num_ineq + test_data.num_eq + test_data.num_lb, :]
                    delta_zu   = test_y[:, test_data.num_var + 2*test_data.num_ineq + test_data.num_eq + test_data.num_lb:, :]
                    
                    # Calculate step sizes. (Ensure that none of the variables are None.)
                    alpha_x, alpha_eta, alpha_s, alpha_zl, alpha_zu = calculate_step(
                        test_data, test_x, delta_x, test_eta, delta_eta, test_s,
                        delta_s, test_zl, delta_zl, test_zu, delta_zu, test_lb, test_ub,
                        args.tau, args.use_line_search, device=DEVICE
                    )

                    # Update variables.
                    test_x = (test_x + alpha_x * delta_x).detach()
                    if test_eta is not None:
                        test_eta = (test_eta + alpha_eta * delta_eta).detach()
                    if test_s is not None:
                        test_s = (test_s + alpha_s * delta_s).detach()
                    if test_lamb is not None:
                        test_lamb = (test_lamb + alpha_x * delta_lamb).detach()
                    if test_zl is not None:
                        test_zl = (test_zl + alpha_zl * delta_zl).detach()
                    if test_zu is not None:
                        test_zu = (test_zu + alpha_zu * delta_zu).detach()
                    
                    test_obj = test_data.obj_fn(test_x, indices=batch["idx"],).mean()
                    test_residual.append(res_norm)
                    test_objs.append(test_obj.detach().cpu().numpy())
                    
                    end_time = time.time()
                    total_time += (end_time - start_time)
                    print(f"Iter: {t_out:3d} | Test_Obj : {test_obj:.3f} | Batch Time : {total_time:.3f} |")
                    
                print("Final Test Objective for current batch: {:.3f}".format(test_obj))
            print("Total test time: {:.3f}".format(total_time))
            print("Mean residual norm: {:.3f}".format(np.mean(test_residual)))
            print("Mean objective value: {:.3f}".format(np.mean(test_objs)))
                    
    else:
        # Fixed-dimension testing branch.
        if args.prob_type == 'QP':
            test_data = optimizee(prob_type=args.prob_type, learning_type='test', num_var=args.num_var, num_ineq=args.num_ineq,
                                  num_eq=args.num_eq, data_size=args.data_size, device=DEVICE)
        else:
            test_data = optimizee(prob_type=args.prob_type, learning_type='test', file_path=file_path, device=DEVICE)
            
        load_path = save_path
        model.load_state_dict(torch.load(load_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            if (test_data.num_lb != 0) and (test_data.num_ub != 0):
                test_x = (test_data.lb + test_data.ub) / 2
            elif (test_data.num_lb != 0) and (test_data.num_ub == 0):
                test_x = test_data.lb + torch.ones(size=test_data.lb.shape, device=args.device)
            elif (test_data.num_lb == 0) and (test_data.num_ub != 0):
                test_x = test_data.ub - torch.ones(size=test_data.ub.shape, device=args.device)
            else:
                test_x = torch.zeros((test_data.Q.shape[0], test_data.num_var, 1), device=args.device)
            print('The number of variables: {}.'.format(test_data.num_var))
            if test_data.num_ineq != 0:
                test_eta = torch.ones((test_data.Q.shape[0], test_data.num_ineq, 1), device=args.device)
                test_s = torch.ones((test_data.Q.shape[0], test_data.num_ineq, 1), device=args.device)
                print('The number of inequalities: {}.'.format(test_data.num_ineq))
            else:
                test_eta = None
                test_s = None
            if test_data.num_eq != 0:
                test_lamb = torch.zeros((test_data.Q.shape[0], test_data.num_eq, 1), device=args.device)
                print('The number of equalities: {}.'.format(test_data.num_eq))
            else:
                test_lamb = None
            if test_data.num_lb != 0:
                test_zl = torch.ones((test_data.Q.shape[0], test_data.num_lb, 1), device=args.device)
                print('The number of lower bounds: {}.'.format(test_data.num_lb))
            else:
                test_zl = None
            if test_data.num_ub != 0:
                test_zu = torch.ones((test_data.Q.shape[0], test_data.num_ub, 1), device=args.device)
                print('The number of upper bounds: {}.'.format(test_data.num_ub))
            else:
                test_zu = None
            test_losses = []
            test_objs = []
            HH_condis = []
            HH_precondis = []
            test_ineq_vio_maxs = []
            test_ineq_vio_means = []
            test_eq_vio_maxs = []
            test_eq_vio_means = []
            test_lb_vio_maxs = []
            test_lb_vio_means = []
            test_ub_vio_maxs = []
            test_ub_vio_means = []
            test_residual = []
            test_y_norms = []
            test_mus = []
            test_F0 = []
            total_time = 0.0
            for t_out in range(args.outer_T):
                start_time = time.time()
                test_J, test_F, test_mu = test_data.cal_kkt(test_x, test_eta, test_s, test_lamb, test_zl, test_zu, args.sigma)
                res_norm = test_F.norm().item()
                print(f"Iter: {t_out:3d} | Residual Norm: {res_norm:8.4f}")
                HH_condis.append(np.linalg.cond(np.array((test_J[0].T @ test_J[0]).cpu().numpy())))
                init_y = torch.zeros((test_data.val_size, test_data.num_var+2*test_data.num_ineq+test_data.num_eq+test_data.num_lb+test_data.num_ub, 1), device=args.device)
                if args.precondi:
                    test_D_values, D_id = (torch.bmm(test_J.permute(0, 2, 1), test_J)).max(-1)
                    test_D_inverse = torch.diag_embed(torch.sqrt(1 / test_D_values))
                    test_J_0 = torch.bmm(test_J, test_D_inverse)
                    HH_precondis.append(np.linalg.cond(np.array((test_J_0[0].T @ test_J_0[0]).cpu().numpy())))
                else:
                    test_J_0 = test_J
                test_y, _, losses = model(test_data, init_y, test_J_0, test_F)
                if args.precondi:
                    test_y = torch.bmm(test_D_inverse, test_y)
                test_losses.append(losses)
                delta_x = test_y[:, :test_data.num_var, :]
                delta_eta = test_y[:, test_data.num_var:test_data.num_var + test_data.num_ineq, :]
                delta_lamb = test_y[:, test_data.num_var + test_data.num_ineq:test_data.num_var + test_data.num_ineq + test_data.num_eq, :]
                delta_s = test_y[:, test_data.num_var + test_data.num_ineq + test_data.num_eq:test_data.num_var + 2 * test_data.num_ineq + test_data.num_eq, :]
                delta_zl = test_y[:, test_data.num_var + 2 * test_data.num_ineq + test_data.num_eq:test_data.num_var + 2 * test_data.num_ineq + test_data.num_eq + test_data.num_lb, :]
                delta_zu = test_y[:, test_data.num_var + 2 * test_data.num_ineq + test_data.num_eq + test_data.num_lb:, :]
                alpha_x, alpha_eta, alpha_s, alpha_zl, alpha_zu = calculate_step(test_data, test_x, delta_x, test_eta, delta_eta,
                                                                                 test_s, delta_s, test_zl, delta_zl, test_zu,
                                                                                 delta_zu, test_data.lb, test_data.ub, args.tau,
                                                                                 args.use_line_search, device=DEVICE)
                if (test_data.num_lb != 0) or (test_data.num_ub != 0):
                    test_x = (test_x + alpha_x * delta_x).detach()
                else:
                    if test_data.num_ineq != 0:
                        test_x = (test_x + alpha_s * delta_x).detach()
                    else:
                        test_x = (test_x + delta_x).detach()
                if test_data.num_ineq != 0:
                    test_eta = (test_eta + alpha_eta * delta_eta).detach()
                    test_s = (test_s + alpha_s * delta_s).detach()
                if test_data.num_eq != 0:
                    if (test_data.num_lb != 0) or (test_data.num_ub != 0):
                        test_lamb = (test_lamb + alpha_x * delta_lamb).detach()
                    else:
                        if test_data.num_ineq != 0:
                            test_lamb = (test_lamb + alpha_s * delta_lamb).detach()
                        else:
                            test_lamb = (test_lamb + delta_lamb).detach()
                if test_data.num_lb != 0:
                    test_zl = (test_zl + alpha_zl * delta_zl).detach()
                if test_data.num_ub != 0:
                    test_zu = (test_zu + alpha_zu * delta_zu).detach()
                end_time = time.time()
                total_time += (end_time-start_time)
                test_residual.append(torch.linalg.vector_norm(torch.bmm(test_J, test_y) + test_F, dim=1).mean().detach().cpu().numpy())
                test_y_norms.append(torch.linalg.vector_norm(test_y, dim=1).mean().detach().cpu().numpy())
                test_mus.append(test_mu.mean().detach().cpu().numpy())
                test_F0.append(torch.linalg.vector_norm(test_data.F0(test_x, test_eta, test_s, test_lamb, test_zl, test_zu, args.sigma), dim=1).mean().detach().cpu().numpy())
                test_obj = test_data.obj_fn(test_x).mean()
                test_objs.append(test_obj.detach().cpu().numpy())
                if test_data.num_ineq != 0:
                    test_ineq_vio_max = test_data.ineq_dist(test_x).max(dim=1).values.mean()
                    test_ineq_vio_mean = test_data.ineq_dist(test_x).mean()
                    test_ineq_vio_maxs.append(test_ineq_vio_max.detach().cpu().numpy())
                    test_ineq_vio_means.append(test_ineq_vio_mean.detach().cpu().numpy())
                if test_data.num_eq != 0:
                    test_eq_vio_max = test_data.eq_dist(test_x).max(dim=1).values.mean()
                    test_eq_vio_mean = test_data.eq_dist(test_x).mean()
                    test_eq_vio_maxs.append(test_eq_vio_max.detach().cpu().numpy())
                    test_eq_vio_means.append(test_eq_vio_mean.detach().cpu().numpy())
                if test_data.num_lb != 0:
                    test_lb_vio_max = test_data.lower_bound_dist(test_x).max(dim=1).values.mean()
                    test_lb_vio_mean = test_data.lower_bound_dist(test_x).mean()
                    test_lb_vio_maxs.append(test_lb_vio_max.detach().cpu().numpy())
                    test_lb_vio_means.append(test_lb_vio_mean.detach().cpu().numpy())
                if test_data.num_ub != 0:
                    test_ub_vio_max = test_data.upper_bound_dist(test_x).max(dim=1).values.mean()
                    test_ub_vio_mean = test_data.upper_bound_dist(test_x).mean()
                    test_ub_vio_maxs.append(test_ub_vio_max.detach().cpu().numpy())
                    test_ub_vio_means.append(test_ub_vio_mean.detach().cpu().numpy())
                print("Iter: {} | Test_Obj : {:.3f} | Test_Time : {:.3f} |".format(t_out, test_obj, total_time))
                if test_data.num_ineq != 0:
                    print("Test_Max_Ineq : {:.3f} | Test_Mean_Ineq : {:.3f} |".format(test_ineq_vio_max, test_ineq_vio_mean))
                if test_data.num_eq != 0:
                    print("Test_Max_Eq : {:.3f} | Test_Mean_Eq : {:.3f} |".format(test_eq_vio_max, test_eq_vio_mean))
                if test_data.num_lb != 0:
                    print("Test_Max_Lb : {:.3f} | Test_Mean_Lb : {:.3f} |".format(test_lb_vio_max, test_lb_vio_mean))
                if test_data.num_ub != 0:
                    print("Test_Max_Ub : {:.3f} | Test_Mean_Ub : {:.3f} |".format(test_ub_vio_max, test_ub_vio_mean))
                best_obj = test_data.obj_fn(test_x).mean()
                print('LSTM Test Objective Value:', best_obj)
            if args.save_sol:
                if args.prob_type == 'Nonconvex_QP':
                    results_save_path = os.path.join(args.save_dir, model.name(), '{}_{}_{}_{}_results.mat'.format(args.prob_type,
                                                                                                                 args.mat_name,
                                                                                                                 args.outer_T,
                                                                                                                 args.inner_T))
                else:
                    results_save_path = os.path.join(args.save_dir, model.name(), '{}_{}_{}_{}_{}_{}_results.mat'.format(args.prob_type,
                                                                                                                            args.num_var,
                                                                                                                            args.num_ineq,
                                                                                                                            args.num_eq,
                                                                                                                            args.outer_T,
                                                                                                                            args.inner_T))
                test_dict = {'time': (total_time),
                             'x': np.array(test_x.detach().cpu()),
                             'inner_losses': np.array(test_losses),
                             'HH_condis': np.array(HH_condis),
                             'HH_precondis':np.array(HH_precondis),
                             'objs': np.array(test_objs),
                             'best_obj': np.array(best_obj.detach().cpu()),
                             'ineq_vio_maxs': np.array(test_ineq_vio_maxs),
                             'ineq_vio_means': np.array(test_ineq_vio_means),
                             'eq_vio_maxs': np.array(test_eq_vio_maxs),
                             'eq_vio_means': np.array(test_eq_vio_means),
                             'residual': np.array(test_residual),
                             'y_norm': np.array(test_y_norms),
                             'mu': np.array(test_mus),
                             'F0': np.array(test_F0)}
                sio.savemat(results_save_path, test_dict)
