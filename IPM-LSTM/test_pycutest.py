from problems.CUTEST import CUTEST as CUTESTDataset
import pycutest
dataset = CUTESTDataset(learning_type='train', file_path='problems/probs/QPset_test.txt')
print("Length of dataset:", len(dataset))
print("Type of first item:", type(dataset[0]))
print("Total problems loaded:", len(dataset.problems))
print("Details for each problem:")
for i, prob in enumerate(dataset.problems):
    print(f"Problem {i}: n = {prob.n}, m = {prob.m}")
    messages = ""
    print(pycutest.print_available_sif_params(prob.name))