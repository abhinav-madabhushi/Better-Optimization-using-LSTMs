import pycutest

problem_name = "ROSENBR"  # Rosenbrock function
problem = pycutest.import_problem(problem_name)

print("Number of variables:", problem.n)
print("Objective function value at x0:", problem.obj(problem.x0))
