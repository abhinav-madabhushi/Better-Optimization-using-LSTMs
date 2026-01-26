Guide
- L20-LSTM-IPM_extended.py is the main file to run, the code for LSTM is in the models folder, and the QP class is in the problems folder.
- Go to the "IPM-LSTM" folder before running any commands.

Note: Due to size constraints, I was not able to push the 90-dimensional dataset to GitHub. To create it, will need to run the following command:

```bash
python random_generator_unconstrained_.py
```

1) GPU Wall Clock for tables

```bash
python testing_graphs.py \
  --ckpt weights/weights_QP_BC_dim10.pt \
  --mat_name QP_convex_10var_0eq_0ineq \

python testing_graphs.py \
  --ckpt weights/weights_QP_BC_dim90_final.pt \
  --mat_name QP_convex_90var_0eq_0ineq \
```

2) Hyperparameter Comparison Graphs

```bash
python L20-LSTM-IPM_extended.py \
  --mode model_comparison \
  --ckpt weights/weights_QP_BC_dim10_hiddensize32_numlayers1.pt \
  --compare_models weights/weights_QP_BC_dim10_hiddensize64_numlayers1.pt weights/weights_QP_BC_dim10_hiddensize128_numlayers1.pt \
 --mat_name QP_convex_10var_0eq_0ineq

python L20-LSTM-IPM_extended.py \
  --mode model_comparison \
  --ckpt weights/weights_QP_BC_dim10_backprop1.pt \
  --compare_models weights/weights_QP_BC_dim10_backprop5.pt weights/weights_QP_BC_dim10_backprop10.pt \
 --mat_name QP_convex_10var_0eq_0ineq
```

3) Performance Comparison with IPOPT Plots
```bash
python L20-LSTM-IPM_extended.py --mode profile --ckpt weights/weights_QP_BC_dim10.pt  --mat_name QP_convex_10var_0eq_0ineq

python L20-LSTM-IPM_extended.py --mode profile --ckpt weights/weights_QP_BC_dim90.pt  --mat_name QP_convex_90var_0eq_0ineq
```

