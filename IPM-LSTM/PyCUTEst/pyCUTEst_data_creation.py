import pycutest
import numpy as np
import configargparse
import scipy.io as sio

# Set up command-line arguments
parser = configargparse.ArgumentParser()
parser.add_argument('--output_prefix', type=str, default='robust_dataset_batch',
                    help='Prefix for the output MAT files.')
parser.add_argument('--min_vars', type=int, default=10,
                    help='Minimum number of decision variables required for a problem.')
parser.add_argument('--min_constraints', type=int, default=1,
                    help='Minimum number of constraints required for a problem.')
parser.add_argument('--batch_size', type=int, default=15,
                    help='Number of problems per batch.')
args = parser.parse_args()

# Get the list of all problems in CUTEst via PyCUTEst.
all_problem_names = pycutest.find_problems()
total_problems = len(all_problem_names)
print(f"Total number of problems in CUTEst: {total_problems}")

# Initialize batch storage lists
batch_problem_names = []
batch_n = []      # number of variables per problem
batch_m = []      # number of constraints per problem
batch_x0 = []     # initial point (n x 1)
batch_f0 = []     # objective value at x0 (scalar)
batch_grad0 = []  # gradient at x0 (n x 1)
batch_Hess0 = []  # Hessian at x0 (n x n)
batch_c0 = []     # constraint values at x0 (m x 1)
batch_lb = []     # lower bounds (n x 1)
batch_ub = []     # upper bounds (n x 1)

batch_count = 0
processed_count = 0

def save_batch(batch_idx, dataset):
    out_file = f"{args.output_prefix}_{batch_idx}.mat"
    sio.savemat(out_file, dataset)
    print(f"Saved batch {batch_idx} with {len(dataset['problem_names'])} problems to {out_file}")

# Loop over all problems
for p_name in all_problem_names:
    try:
        # Import the problem.
        problem = pycutest.import_problem(p_name)
        n = problem.n
        m = problem.m if hasattr(problem, 'm') else 0
        
        # Filter: only include problems with at least min_vars and min_constraints.
        if n < args.min_vars or m < args.min_constraints:
            continue
        
        # Extract initial point.
        x0 = problem.x0  # shape (n,)
        
        # Evaluate objective and gradient at x0.
        f0 = problem.obj(x0)
        grad0 = np.array(problem.grad(x0)).reshape(n, 1)
        
        # Try to get Hessian; if not available, use identity.
        try:
            Hess0 = np.array(problem.hess(x0))
        except Exception as e:
            Hess0 = np.eye(n)
        
        # Get constraint function values.
        try:
            c0 = np.array(problem.cons(x0)).reshape(m, 1)
        except Exception as e:
            print(f"Skipping {p_name}: cannot evaluate constraints.")
            continue
        
        # Extract variable bounds if provided; else use defaults.
        if hasattr(problem, 'lb') and problem.lb is not None:
            lb = np.array(problem.lb).reshape(n, 1)
        else:
            lb = -np.inf * np.ones((n, 1))
        if hasattr(problem, 'ub') and problem.ub is not None:
            ub = np.array(problem.ub).reshape(n, 1)
        else:
            ub = np.inf * np.ones((n, 1))
        
        # Append the extracted data to the current batch.
        batch_problem_names.append(p_name)
        batch_n.append(n)
        batch_m.append(m)
        batch_x0.append(x0.reshape(n, 1))
        batch_f0.append(f0)
        batch_grad0.append(grad0)
        batch_Hess0.append(Hess0)
        batch_c0.append(c0)
        batch_lb.append(lb)
        batch_ub.append(ub)
        
        processed_count += 1
        
        # If we've reached the batch size, save this batch and reset lists.
        if len(batch_problem_names) >= args.batch_size:
            batch_count += 1
            dataset = {
                'problem_names': np.array(batch_problem_names, dtype=object),
                'n': np.array(batch_n),
                'm': np.array(batch_m),
                'x0': batch_x0,
                'f0': np.array(batch_f0),
                'grad0': batch_grad0,
                'Hess0': batch_Hess0,
                'c0': batch_c0,
                'lb': batch_lb,
                'ub': batch_ub
            }
            save_batch(batch_count, dataset)
            # Reset batch lists
            batch_problem_names = []
            batch_n = []
            batch_m = []
            batch_x0 = []
            batch_f0 = []
            batch_grad0 = []
            batch_Hess0 = []
            batch_c0 = []
            batch_lb = []
            batch_ub = []
            
    except Exception as e:
        print(f"Skipping problem {p_name} due to error: {e}")
        continue

# Save any remaining problems in the last batch.
if batch_problem_names:
    batch_count += 1
    dataset = {
        'problem_names': np.array(batch_problem_names, dtype=object),
        'n': np.array(batch_n),
        'm': np.array(batch_m),
        'x0': batch_x0,
        'f0': np.array(batch_f0),
        'grad0': batch_grad0,
        'Hess0': batch_Hess0,
        'c0': batch_c0,
        'lb': batch_lb,
        'ub': batch_ub
    }
    save_batch(batch_count, dataset)

print(f"Total problems processed: {processed_count}")
