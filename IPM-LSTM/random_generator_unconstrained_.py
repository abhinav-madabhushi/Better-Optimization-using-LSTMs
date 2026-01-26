# import numpy as np
# import scipy.io as sio
# import os

# # parameters
# N = 10      # number of problems
# n = 1000          # num variables
# out_path = 'datasets/qp/unconstrained_QP_dim_1000.mat'
# os.makedirs(os.path.dirname(out_path), exist_ok=True)

# # 1) Q and p as before
# diag_entries = np.random.random((N, n))
# Q = np.zeros((N, n, n))
# for var in range(n):
#     Q[:, var, var] = diag_entries[:, var]
# p = np.random.random((N, n))

# # # Sample once
# # diag_entries = np.random.random(n)        # shape [n], constant
# # p_single     = np.random.random(n)        # shape [n], constant

# # # Broadcast to all N problems
# # Q = np.zeros((N, n, n))
# # Q[:, np.arange(n), np.arange(n)] = diag_entries  # same diagonal for every problem
# # p = np.tile(p_single, (N, 1))                    # same p for every problem


# # 2) init all bounds as “unbounded”
# lb = np.full((N, n), -np.inf)
# ub = np.full((N, n),  np.inf)

# # 3) pick a type for each problem
# probs = [0, 0, 1, 0, 0]                    # lower, upper, both, partial, none
# types = np.random.choice(['lower','upper','both','partial','none'], size=N, p=probs)

# for i, t in enumerate(types):
#     if t == 'lower':
#         # both variables get a lower bound
#         lb[i] = np.random.uniform(-1, 1, size=n)

#     elif t == 'upper':
#         # both variables get an upper bound
#         ub[i] = np.random.uniform(-1, 1, size=n)

#     elif t == 'both':
#         # each variable gets both bounds; ensure lb<=ub by sorting two draws
#         a = np.random.uniform(-1, 1, size=n)
#         b = np.random.uniform(-1, 1, size=n)
#         lb[i] = np.minimum(a, b)
#         ub[i] = np.maximum(a, b)

#     elif t == 'partial':
#         # pick one variable at random
#         j = np.random.randint(n)
#         # pick lower *or* upper with 50/50 chance
#         if np.random.rand() < 0.5:
#             lb[i, j] = np.random.uniform(-1, 1)
#         else:
#             ub[i, j] = np.random.uniform(-1, 1)
#         # the other var remains unbounded
#     else:  # 'none' -> keep lb/ub as +/- inf
#         pass

# # 4) save everything
# sio.savemat(
#     out_path,
#     {
#         'Q':  Q,   # (N,2,2)
#         'p':  p,   # (N,2)
#         'lb': lb,  # (N,2)
#         'ub': ub   # (N,2)
#     }
# )
# print(f"Saved {out_path} with {N} mixed-type 2-var QPs.")

# import numpy as np
# import scipy.io as sio
# import os

# # parameters
# N = 10_000                     # number of problems
# n_min, n_max = 5, 10         # true dimension range (inclusive)
# out_path = 'datasets/qp/unconstrained_QP_variable_dim_5_10.mat'
# os.makedirs(os.path.dirname(out_path), exist_ok=True)

# # preallocate padded arrays (numeric, not object)
# Q  = np.zeros((N, n_max, n_max), dtype=np.float32)     # PSD diagonal Q, padded with 0
# p  = np.zeros((N, n_max), dtype=np.float32)            # padded with 0
# lb = np.full((N, n_max), -np.inf, dtype=np.float32)    # padded with -inf
# ub = np.full((N, n_max),  +np.inf, dtype=np.float32)   # padded with +inf
# n_vec = np.empty((N,), dtype=np.int32)                 # true per-problem dimensions

# # (optional) encode bound type as integers to keep .mat fully numeric
# # 0=lower, 1=upper, 2=both, 3=partial
# type_id = np.empty((N,), dtype=np.int32)

# # distribution over bound types (edit as you like)
# probs = [0.00, 0.00, 1.00, 0.00]  # here: always 'both'

# for i in range(N):
#     n_i = np.random.randint(n_min, n_max + 1)
#     n_vec[i] = n_i

#     # PSD diagonal Q with positive diagonal
#     diag_entries = np.random.random(n_i).astype(np.float32)  # in (0,1)
#     # write diagonal directly into the padded Q block
#     idx = np.arange(n_i)
#     Q[i, idx, idx] = diag_entries

#     # linear term
#     p[i, :n_i] = np.random.random(n_i).astype(np.float32)

#     # bounds
#     t = np.random.choice(4, p=probs)  # 0=lower,1=upper,2=both,3=partial
#     type_id[i] = t

#     if t == 0:  # lower
#         lb[i, :n_i] = np.random.uniform(-1, 1, size=n_i).astype(np.float32)
#     elif t == 1:  # upper
#         ub[i, :n_i] = np.random.uniform(-1, 1, size=n_i).astype(np.float32)
#     elif t == 2:  # both
#         a = np.random.uniform(-1, 1, size=n_i).astype(np.float32)
#         b = np.random.uniform(-1, 1, size=n_i).astype(np.float32)
#         lb[i, :n_i] = np.minimum(a, b)
#         ub[i, :n_i] = np.maximum(a, b)
#     else:  # partial
#         j = np.random.randint(n_i)
#         if np.random.rand() < 0.5:
#             lb[i, j] = np.random.uniform(-1, 1)
#         else:
#             ub[i, j] = np.random.uniform(-1, 1)

# print(n_vec.shape)
# # save: fully numeric arrays (no object dtypes)
# sio.savemat(
#     out_path,
#     {
#         'Q':  Q,            # (N, 100, 100) float32
#         'p':  p,            # (N, 100)      float32
#         'lb': lb,           # (N, 100)      float32, -inf padded
#         'ub': ub,           # (N, 100)      float32, +inf padded
#         'n':  n_vec,        # (N,)          int32   (true dims)
#         'type_id': type_id  # (N,)          int32   (bound type)
#     },
#     do_compression=True,
# )

# print(f"Saved {out_path} with {N} QPs padded to {n_max}D (true n in [{n_min},{n_max}]).")

# import numpy as np
# import scipy.io as sio
# import os

# # parameters
# N = 10000      # number of problems
# n = 10          # num variables
# out_path = 'datasets/qp/unconstrained_QP_nonconvex_full_Q_dim_10_lbubrandom.mat'
# os.makedirs(os.path.dirname(out_path), exist_ok=True)
# rng = np.random.default_rng(42)

# # Per-problem type: "convex" (SPD) vs "nonconvex" (indefinite)
# types = np.where(rng.random(N) < 0.5, "convex", "nonconvex")

# # Eigenvalue magnitude ranges
# m, L = 1e-3, 10.0          # positive eigenvalues in [m, L] for convex branch
# mneg, Lneg = 1e-3, 10.0    # negative magnitudes in [mneg, Lneg] for nonconvex branch
# frac_negative = 0.3        # fraction of negative eigenvalues in nonconvex branch

# # --- Build batch of symmetric Q (non-diagonal) ---
# Q = np.empty((N, n, n), dtype=float)

# for i in range(N):
#     # Random orthonormal basis via QR (gives non-diagonal Q generically)
#     A = rng.standard_normal((n, n))
#     Ui, R = np.linalg.qr(A)
#     # Make QR orientation consistent
#     d = np.sign(np.diag(R))
#     Ui = Ui * d

#     if types[i] == "convex":
#         # SPD: all positive eigenvalues
#         lam = rng.uniform(m, L, size=n)
#     else:
#         # Indefinite: mix of negative and positive eigenvalues
#         k_neg = max(1, int(round(frac_negative * n)))
#         idx = rng.permutation(n)
#         neg_idx, pos_idx = idx[:k_neg], idx[k_neg:]

#         lam = np.empty(n, dtype=float)
#         lam[neg_idx] = -rng.uniform(mneg, Lneg, size=k_neg)

#         if len(pos_idx) == 0:
#             # Ensure at least one positive eigenvalue
#             pos_idx = np.array([neg_idx[-1]])
#             lam[pos_idx] = rng.uniform(m, L, size=1)
#             lam[neg_idx[-1]] = -rng.uniform(mneg, Lneg)
#         else:
#             lam[pos_idx] = rng.uniform(m, L, size=len(pos_idx))

#     # Form Q = U diag(lam) U^T without explicitly building diag:
#     Qi = (Ui * lam) @ Ui.T
#     Qi = 0.5 * (Qi + Qi.T)  # numerical symmetrization
#     Q[i] = Qi

# p = np.random.random((N, n))

# # 2) init all bounds as “unbounded”
# lb = np.full((N, n), -np.inf)
# ub = np.full((N, n),  np.inf)

# # 3) pick a type for each problem
# probs = [0.2, 0.2, 0.2, 0.4]                    # lower, upper, both, partial
# types = np.random.choice(['lower','upper','both','partial'], size=N, p=probs)

# # 3) pick a type for each problem
# probs = [0.2, 0.2, 0.2, 0.4]  # lower, upper, both, partial
# types = rng.choice(['lower', 'upper', 'both', 'partial'], size=N, p=probs)

# for i, t in enumerate(types):
#     if t == 'lower':
#         lb[i] = rng.uniform(-1, 1, size=n)

#     elif t == 'upper':
#         ub[i] = rng.uniform(-1, 1, size=n)

#     elif t == 'both':
#         a = rng.uniform(-1, 1, size=n)
#         b = rng.uniform(-1, 1, size=n)
#         lb[i] = np.minimum(a, b)
#         ub[i] = np.maximum(a, b)

#     else:  # 'partial' → bound a random subset (size 1..⌊n/2⌋) with mixed lower/upper
#         kmax = max(1, n // 2)
#         k = rng.integers(1, kmax + 1)
#         idx = rng.choice(n, size=k, replace=False)

#         choose_lower = rng.random(k) < 0.5
#         if choose_lower.any():
#             lb[i, idx[choose_lower]] = rng.uniform(-1, 1, size=choose_lower.sum())
#         if (~choose_lower).any():
#             ub[i, idx[~choose_lower]] = rng.uniform(-1, 1, size=(~choose_lower).sum())


# # 4) save everything
# sio.savemat(
#     out_path,
#     {
#         'Q':  Q,   # (N,2,2)
#         'p':  p,   # (N,2)
#         'lb': lb,  # (N,2)
#         'ub': ub   # (N,2)
#     }
# )
# print(f"Saved {out_path} with {N} mixed-type 2-var QPs.")

import numpy as np
import scipy.io as sio
import os

# ---------------------- Parameters ----------------------
N = 10000        # number of problems
n = 90          # number of variables
num_ineq = 0
num_eq = 0
out_path = 'datasets/qp/QP_convex_90var_0eq_0ineq.mat'
os.makedirs(os.path.dirname(out_path), exist_ok=True)

seed = 42
rng = np.random.default_rng(seed)

# Quadratic form spectrum ranges
m, L = 1e-3, 10.0          # positive eigenvalues for convex branch
mneg, Lneg = 1e-3, 10.0    # negative magnitudes for nonconvex branch
frac_negative = 0.3        # fraction of negative eigenvalues in nonconvex branch

# Probability a problem is convex vs nonconvex
prob_convex = 1.0  # 1.0 = always convex

# Box-bound scheme probabilities: ['lower','upper','both','partial','none']
box_probs = [0, 0, 1, 0, 0]  # always 'both' in this config; change as desired

#box_probs = [0, 0, 1, 0] 

# Slack margin for inequalities: G x0 - c = s >= slack_min
slack_min = 0.1
slack_max = 1.0

# Scale of random A, G rows
row_scale_A = 1.0
row_scale_G = 1.0

# --------------------------------------------------------

# --- 1) Build batch of symmetric Q (non-diagonal) ---
Q = np.empty((N, n, n), dtype=float)

# Per-problem convex/nonconvex flag
qc_types = np.where(rng.random(N) < prob_convex, "convex", "nonconvex")

for i in range(N):
    # Random orthonormal basis via QR (non-diagonal Q)
    Aqr = rng.standard_normal((n, n))
    Ui, R = np.linalg.qr(Aqr)
    # Consistent orientation
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Ui = Ui * d

    if qc_types[i] == "convex":
        lam = rng.uniform(m, L, size=n)
    else:
        k_neg = max(1, int(round(frac_negative * n)))
        idx = rng.permutation(n)
        neg_idx, pos_idx = idx[:k_neg], idx[k_neg:]
        lam = np.empty(n, dtype=float)
        lam[neg_idx] = -rng.uniform(mneg, Lneg, size=k_neg)
        if len(pos_idx) == 0:
            pos_idx = np.array([neg_idx[-1]])
            lam[pos_idx] = rng.uniform(m, L, size=1)
            lam[neg_idx[-1]] = -rng.uniform(mneg, Lneg)
        else:
            lam[pos_idx] = rng.uniform(m, L, size=len(pos_idx))

    Qi = (Ui * lam) @ Ui.T
    Qi = 0.5 * (Qi + Qi.T)  # numerical symmetrization
    Q[i] = Qi

# Linear term
p = rng.random((N, n))

# --- 2) Box bounds (unbounded by default) ---
lb = np.full((N, n), -np.inf, dtype=float)
ub = np.full((N, n),  np.inf, dtype=float)

box_types = rng.choice(['lower', 'upper', 'both', 'partial', 'none'], size=N, p=box_probs)

for i, t in enumerate(box_types):
    if t == 'lower':
        lb[i] = rng.uniform(-1, 1, size=n)
    elif t == 'upper':
        ub[i] = rng.uniform(-1, 1, size=n)
    elif t == 'both':
        a = rng.uniform(-1, 1, size=n)
        b = rng.uniform(-1, 1, size=n)
        lb[i] = np.minimum(a, b)
        ub[i] = np.maximum(a, b)
    elif t == 'partial':
        p_fin_lb = 0.5
        p_fin_ub = 0.5

        fin_lb_mask = rng.random(n) < p_fin_lb   # True → finite lb
        fin_ub_mask = rng.random(n) < p_fin_ub   # True → finite ub

        lb_i = np.full(n, -np.inf, dtype=float)
        ub_i = np.full(n,  np.inf, dtype=float)

        # assign random finite bounds where selected
        if fin_lb_mask.any():
            lb_i[fin_lb_mask] = rng.uniform(-1, 1, size=fin_lb_mask.sum())
        if fin_ub_mask.any():
            ub_i[fin_ub_mask] = rng.uniform(-1, 1, size=fin_ub_mask.sum())

        # fix any coordinates where both finite but lb > ub
        both_fin = fin_lb_mask & fin_ub_mask
        bad = both_fin & (lb_i > ub_i)
        if bad.any():
            tmp = lb_i[bad].copy()
            lb_i[bad] = ub_i[bad]
            ub_i[bad] = tmp

        lb[i] = lb_i
        ub[i] = ub_i
    else:  # 'none'
        # leave lb/ub as +/- inf
        pass

# --- helper to draw x0 respecting possibly-infinite bounds ---
def sample_x0(lb_i, ub_i):
    x0 = np.zeros_like(lb_i)
    for j in range(lb_i.size):
        lo, hi = lb_i[j], ub_i[j]
        if np.isfinite(lo) and np.isfinite(hi):
            if lo > hi:
                lo, hi = hi, lo
            if hi - lo < 1e-12:
                x0[j] = lo
            else:
                x0[j] = rng.uniform(lo, hi)
        elif np.isfinite(lo) and not np.isfinite(hi):
            x0[j] = lo + abs(rng.normal(0.0, 0.5))
        elif not np.isfinite(lo) and np.isfinite(hi):
            x0[j] = hi - abs(rng.normal(0.0, 0.5))
        else:
            x0[j] = rng.normal(0.0, 1.0)
    eps = 1e-6
    x0 = np.maximum(x0, np.where(np.isfinite(lb_i), lb_i + eps, x0))
    x0 = np.minimum(x0, np.where(np.isfinite(ub_i), ub_i - eps, x0))
    return x0

# --- 3) Fixed-size A, b and G, c using num_eq / num_ineq ---
A = np.zeros((N, num_eq,   n), dtype=float)
b = np.zeros((N, num_eq,   1), dtype=float)
G = np.zeros((N, num_ineq, n), dtype=float)
c = np.zeros((N, num_ineq, 1), dtype=float)

for i in range(N):
    # Feasible point
    x0 = sample_x0(lb[i], ub[i])

    # Equalities: A_i x0 = b_i
    if num_eq > 0:
        Ai = row_scale_A * rng.normal(0.0, 1.0, size=(num_eq, n))
        bi = (Ai @ x0).reshape(num_eq, 1)
        A[i] = Ai
        b[i] = bi

    # Inequalities: G_i x0 - c_i = s_i >= slack_min
    if num_ineq > 0:
        Gi = row_scale_G * rng.normal(0.0, 1.0, size=(num_ineq, n))
        s  = rng.uniform(slack_min, slack_max, size=(num_ineq, 1))
        ci = (Gi @ x0).reshape(num_ineq, 1) - s
        G[i] = Gi
        c[i] = ci

# 4) Save everything
sio.savemat(
    out_path,
    {
        'Q':  Q,             # (N, n, n)
        'p':  p,             # (N, n)
        'lb': lb,            # (N, n)
        'ub': ub,            # (N, n)
        'G':  G,             # (N, num_ineq, n)
        'c':  c,             # (N, num_ineq, 1)
        'A':  A,             # (N, num_eq,   n)
        'b':  b,             # (N, num_eq,   1)
        # 'qc_types':  qc_types,   # (N,) "convex" / "nonconvex"
        # 'box_types': box_types   # (N,) 'lower'/'upper'/'both'/'partial'
    }
)

print(f"Saved {out_path} with {N} problems: n={n}, eq={num_eq}, ineq={num_ineq}.")


# # portfolio optimization problem generation
# import numpy as np
# import scipy.io as sio
# import os

# N = 10000          # number of problems
# s = 50              # dim of x (assets)
# t = 5               # dim of y (factors)
# n = s + t           # total variables in z = [x; y]
# out_path = f'datasets/qp/portfolio_QP_box_s{s}_t{t}_eq{t+1}.mat'
# os.makedirs(os.path.dirname(out_path), exist_ok=True)

# seed = 42
# rng = np.random.default_rng(seed)

# # D diagonal entries ~ U(0, sqrt(t))  (non-negative)
# D_low, D_high = 0.0, np.sqrt(t)

# # F has exactly K nonzeros, positions random, values ~ N(0,1)
# K_nonzero_F = 25

# # ---------------------- Allocate ------------------------
# Q = np.zeros((N, n, n), dtype=float)
# p = np.zeros((N, n),     dtype=float)

# # Box bounds: encode x>=0 via lb; y free
# lb = np.full((N, n), -np.inf, dtype=float)
# ub = np.full((N, n),  np.inf, dtype=float)

# # Equalities: t rows for y - Fx = 0, plus one row for 1^T x = 1
# num_eq = t + 1
# A = np.zeros((N, num_eq, n), dtype=float)
# b = np.zeros((N, num_eq, 1), dtype=float)

# # ---------------------- Build ---------------------------
# for i in range(N):
#     # --- D (s×s, diagonal) ---
#     D_diag = rng.uniform(D_low, D_high, size=s)
#     D = np.diag(D_diag)

#     # --- F (t×s), exactly K_nonzero_F nonzeros ~ N(0,1) ---
#     F = np.zeros((t, s))
#     k = min(K_nonzero_F, t * s)
#     flat_idx = rng.choice(t * s, size=k, replace=False)
#     rows = flat_idx // s
#     cols = flat_idx % s
#     F[rows, cols] = rng.normal(0.0, 1.0, size=k)

#     # --- mu ~ N(0,1) over x only ---
#     mu = rng.normal(0.0, 1.0, size=s)

#     # --- Assemble Q (n×n) and p (n,) with z=[x;y] order ---
#     Qi = np.zeros((n, n))
#     Qi[:s, :s] = D
#     Qi[s:, s:] = np.eye(t)               # 0.5 * y^T y
#     Q[i] = Qi

#     pi = np.zeros(n)
#     pi[:s] = -mu                         # -mu^T x
#     p[i] = pi

#     # --- Equalities: [ -F  I_t ] [x;y] = 0  and  [1^T  0] [x;y] = 1 ---
#     Ai = np.zeros((num_eq, n))
#     Ai[:t, :s] = -F
#     Ai[:t, s:] = np.eye(t)
#     Ai[t, :s]  = 1.0                     # sum(x)=1
#     bi = np.zeros((num_eq, 1))
#     bi[t, 0] = 1.0
#     A[i] = Ai
#     b[i] = bi

#     # --- Box bounds for x: x >= 0; y free ---
#     lb[i, :s] = 0.0                      # x >= 0
#     # (ub remains +inf everywhere)

# sio.savemat(
#     out_path,
#     {
#         'Q':  Q,             # (N, n, n)
#         'p':  p,             # (N, n)
#         'lb': lb,            # (N, n)  with lb[:, :s] = 0
#         'ub': ub,            # (N, n)  (+inf)
#         'A':  A,             # (N, t+1, n)
#         'b':  b,             # (N, t+1, 1)
#     }
# )

# print(f"Saved {out_path} with {N} portfolio QPs: n={n} (s={s}, t={t}), eq={num_eq}, box x>=0.")
