#!/usr/bin/env python3
import os, sys
# Set environment variables before any other output
os.environ["CUTEST"] = "/home/abhinavm/cutest/cutest"
os.environ["MYARCH"] = "pc64.lnx.gfo"
os.environ["SIFDECODE"] = "/home/abhinavm/cutest/sifdecode"
os.environ["MASTSIF"] = "/home/abhinavm/cutest/mastsif"

# Debug prints to stderr only
print("DEBUG: CUTEST =", os.environ.get("CUTEST"), file=sys.stderr)
print("DEBUG: MYARCH =", os.environ.get("MYARCH"), file=sys.stderr)

import argparse
import json
import pycutest
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Load a list of CUTEST problems and extract data")
    parser.add_argument('--problem_list', type=str, required=True, help="Path to a text file with CUTEST problem names")
    return parser.parse_args()

def finite_diff_jac(func, x, eps=1e-8):
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

def main():
    args = parse_args()
    with open(args.problem_list, 'r') as f:
        problems = [ln.strip() for ln in f if ln.strip()]

    dataset = []
    for name in problems:
        try:
            prob = pycutest.import_problem(name)
            n = prob.n
            x0 = np.array(prob.x0)
            grad0 = prob.grad(x0)
            try:
                Hess0 = prob.hess(x0)
            except Exception:
                Hess0 = np.eye(n)
            
            m = prob.m
            if m is None:
                m = 0
            if m > 0:
                cvals = prob.cons(x0)
                J = finite_diff_jac(prob.cons, x0)
                half_ineq = m // 2
                half_eq = m - half_ineq
                G = J[:half_ineq, :]
                c = cvals[:half_ineq]
                A = J[half_ineq:, :]
                b = cvals[half_ineq:]
            else:
                G, c = np.zeros((0,n)), np.zeros(0)
                A, b = np.zeros((0,n)), np.zeros(0)
            
            lb = prob.lb if getattr(prob, 'lb', None) is not None else -np.inf * np.ones(n)
            ub = prob.ub if getattr(prob, 'ub', None) is not None else np.inf * np.ones(n)

            item = {
                "problem_name": name,
                "n": int(n),
                "Q": Hess0.tolist(),
                "p": grad0.tolist(),
                "G": G.tolist(),
                "c": c.tolist(),
                "A": A.tolist(),
                "b": b.tolist(),
                "lb": lb.tolist(),
                "ub": ub.tolist()
            }
            dataset.append(item)
        except Exception as e:
            print(f"[ERROR] Skipping problem {name} due to error: {e}", file=sys.stderr)
            continue

    # Print JSON to stdout
    output = json.dumps(dataset)
    print(output)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
