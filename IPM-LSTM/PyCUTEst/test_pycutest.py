from problems.CUTEST import CUTEST as CUTESTDataset
dataset = CUTESTDataset(learning_type='train', file_path='/mnt/c/Users/krish/abhinav/Better Optimization using LSTMs/IPM-LSTM/problems/probs/QPset.txt')
print("Length of dataset:", len(dataset))
print("Type of first item:", type(dataset[0]))
