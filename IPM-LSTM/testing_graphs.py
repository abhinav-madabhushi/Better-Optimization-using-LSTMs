import argparse
import csv
import os
import copy
import subprocess
import time

import numpy as np
import torch
from importlib.machinery import SourceFileLoader
from pathlib import Path

from problems.QP_extended import QP

# load the extended IPM module dynamically (filename contains a dash)
L20_EXT_PATH = Path(__file__).resolve().parent / "L20-LSTM-IPM_extended.py"
l20_mod = SourceFileLoader("l20_ext", str(L20_EXT_PATH)).load_module()
PS_L20_LSTM = l20_mod.PS_L20_LSTM
solve_one_qcqp = l20_mod.solve_one_qcqp
K_INNER = l20_mod.K_INNER
LR_META = l20_mod.LR_META


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
    mini = pool.__class__.__new__(pool.__class__)
    mini.__dict__ = dict(pool.__dict__)
    for name, val in pool.__dict__.items():
        if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.size(0) == N:
            mini.__dict__[name] = val.index_select(0, idxs)
    return mini


def time_learned_optimizer(test_pool, net, backprop_every, device, batch_size):
    net.eval()
    dummy_opt = torch.optim.Adam(net.parameters(), lr=LR_META)
    perm = torch.arange(test_pool.Q.shape[0], device=test_pool.Q.device)
    total_time = 0.0
    iterations = []
    with torch.no_grad():
        for i in range(0, len(perm), batch_size):
            idxs = perm[i : i + batch_size].to(device).long()
            mini = slice_pool(test_pool, idxs, len(perm))
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            iters = solve_one_qcqp(
                dummy_opt,
                mini,
                net,
                train_test="test",
                return_info=False,
                backprop_every=backprop_every,
            )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_time += time.perf_counter() - t0
            iterations.append(iters if isinstance(iters, (int, float)) else 0)
    avg_iters = sum(iterations) / len(iterations) if iterations else 0.0
    return total_time, avg_iters


def time_ipopt(test_pool, tol=1e-2):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    _, _, para_times, ip_iters, _ = test_pool.opt_solve(
        solver_type="ipopt_box_qp_extended",
        tol=tol,
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    if para_times is None:
        para_times = []
    if isinstance(para_times, (float, int)):
        total_time = float(para_times)
    else:
        try:
            total_time = float(np.sum(para_times))
        except Exception:
            total_time = 0.0
    return total_time, ip_iters


def time_main_py(cmd, workdir):
    t0 = time.perf_counter()
    res = subprocess.run(cmd, shell=True, cwd=workdir, capture_output=True, text=True)
    total = time.perf_counter() - t0
    val_time = None
    if res.stdout:
        for line in res.stdout.splitlines():
            if "Val_Time" in line:
                try:
                    parts = line.split("Val_Time")
                    tail = parts[-1]
                    val_time = float(tail.split("|")[0].split(":")[-1])
                    break
                except Exception:
                    val_time = None
    return total, res.returncode, val_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to learned optimizer weights.")
    parser.add_argument("--mat_name", required=True, help="Dataset name (without .mat).")
    parser.add_argument("--device", default=None, help="Preferred device for learned optimizer/main.py (default: cuda if available else cpu).")
    parser.add_argument("--backprop", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--output_csv", default=None, help="Output CSV filename (saved under Plots/QP_BC). Defaults to timings_{mat_name}.csv.")
    parser.add_argument(
        "--main_cmd",
        default='python main.py --config ./configs/QP.yaml --prob_type QP_RHS --device {device} --mat_name "{mat}" --inner_T 5 --outer_T 10 --num_epoch 1',
        help="Command template to run main.py timing; must include {mat} and {device} placeholders.",
    )
    args = parser.parse_args()

    device_learned = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_main_flag = "cuda" if device_learned.type == "cuda" else "cpu"

    file_path = os.path.join("datasets", "qp", f"{args.mat_name}.mat")
    pool_cpu = QP(prob_type="QP_unconstrained", learning_type="val", file_path=file_path, seed=17)
    N = move_pool_to_device(pool_cpu, torch.device("cpu"))
    pool_learned = copy.copy(pool_cpu)
    for name, val in pool_cpu.__dict__.items():
        if isinstance(val, torch.Tensor):
            pool_learned.__dict__[name] = val.to(device_learned)

    net = PS_L20_LSTM(
        problem=pool_learned,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        K_inner=K_INNER,
        device=device_learned,
    ).to(device_learned)
    net.load_state_dict(torch.load(args.ckpt, map_location=device_learned))

    # Learned optimizer timing (GPU if available)
    learned_time, learned_iters = time_learned_optimizer(
        pool_learned, net, args.backprop, device_learned, args.batch_size
    )

    # Ipopt timing (force CPU pool)
    ipopt_time, ip_iters = time_ipopt(pool_cpu)

    # main.py timing (GPU if available)
    cmd = args.main_cmd.format(mat=args.mat_name, device=device_main_flag)
    main_time, main_rc, main_val_time = time_main_py(cmd, workdir=os.path.dirname(__file__) or ".")
    # second main.py run with inner_T=10, outer_T=100
    cmd2 = args.main_cmd.replace("--inner_T 5 --outer_T 10", "--inner_T 10 --outer_T 100")
    main2_time, main2_rc, main2_val_time = time_main_py(cmd2.format(mat=args.mat_name, device=device_main_flag), workdir=os.path.dirname(__file__) or ".")

    # write csv
    rows = [
        {"solver": "learned", "val_time_s": learned_time, "avg_iters": learned_iters, "mat_name": args.mat_name},
        {"solver": "ipopt", "val_time_s": ipopt_time, "avg_iters": ip_iters, "mat_name": args.mat_name},
        {"solver": "IPM-LSTM(1)", "val_time_s": main_val_time if main_val_time is not None else main_time, "avg_iters": 50, "mat_name": args.mat_name},
        {"solver": "IPM-LSTM(2)", "val_time_s": main2_val_time if main2_val_time is not None else main2_time, "avg_iters": 1000, "mat_name": args.mat_name},
    ]
    header = ["solver", "val_time_s", "avg_iters", "mat_name"]
    out_dir = Path("Plots") / "QP_BC"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / (args.output_csv or f"timings_{args.mat_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved timings to {csv_path}")


if __name__ == "__main__":
    main()
