import ot
import otdd
import numpy as np
from otdd.pytorch.distance import DatasetDistance
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import argparse
import sys
#
import math
sys.path.append("/external1/nguyenpham/code/uncertainty_shapley/our_method")
from my_gpytorch import settings
from my_gpytorch import utils
from my_gpytorch.kernels import *
from my_gpytorch.mymodels import *
from uncertainty_framework import uncertainty_framework
from my_gpytorch.dataset import *
from my_gpytorch.mymodels import *
from net import *
import pdb
import dill
from joblib import Parallel, delayed
import multiprocessing
import time
model_dict = {
    "Net": Net, 
    "MLPRegressor": MLPRegressor,
    "ResNet_18_Classifier": ResNet_18_Classifier, 
    "CNNRegressor": CNNRegressor
    
}
kernel_dict = {
    "My_OTDD_SW_Kernel": My_OTDD_SW_Kernel,
    "Exponential_SW_Kernel": Exponential_SW_Kernel, 
    "base": RBFKernel
}
def load_data(data_path, options={}):
    with open(data_path, 'rb') as file:
        data_bytes = file.read()
        ds = dill.loads(data_bytes)
    print(ds.keys())
    v_values = ds.get("v_value", None)
    if v_values is not None:
        v_values = torch.tensor(v_values)
    X_train = ds["X_train"]
    y_train = np.array(ds["y_train"])
    X_val = ds["X_val"]
    y_val = np.array(ds["y_val"])
    idx_party = ds["idx_party"]
    dataset = []
    for idx in np.unique(idx_party):
        ids = My_Single_Dataset(torch.tensor(X_train[idx_party == idx]), torch.tensor(y_train[idx_party == idx]))
        dataset.append(ids)
    n_parties = len(np.unique(idx_party))
    D_test = My_Single_Dataset(torch.tensor(X_val), torch.tensor(y_val))
    D_test = torch.utils.data.DataLoader(D_test, 
        batch_size=8,
        pin_memory=True, 
        shuffle=False)
    coalitions = torch.tensor(utils.possible_coalitions(n_parties))
    return dataset, D_test, coalitions, v_values


def __main__(args):
    data_path = args.data
    dataset, D_test, coalitions, v_values = load_data(data_path)
    output_folder = args.out_dir
    output_name   = data_path.split("/")[-1]
    output_name = output_name.split(".")[0]
    # 
    model_aggregate = args.model_aggregate
    n_random = args.n_random
    n_active = args.n_active
    kernel = kernel_dict[args.kernel]
    embding_func = None
    args.device = "cuda:{}".format(args.device) if torch.cuda.is_available() else 'cpu'
    if len(args.device_ids) == 0:
        args.device_ids = None
    if args.embd:
        embding_func = MBS_mapping_label
    update_interval = None
    if args.update_interval > 0:
        update_interval = args.update_interval

    n_embed = args.output_dim
    if n_embed == 0:
        embding_func = None
    aggregate_model = model_dict[args.model_aggregate]
    print(args, embding_func, kernel, model_aggregate)

    #
    output_path = os.path.join(output_folder,
                          "{}_{}_r{}_a{}_seed{}_ey{}_eta_{}.pickle".format(output_name, args.kernel, n_random, n_active, args.seed, n_embed, args.eta))
    f_model_path = os.path.join(output_folder,"f_model{}_{}_r{}_a{}_seed{}_ey{}.ph".format(output_name, args.kernel, n_random, n_active, args.seed, n_embed))
    print(args, embding_func)
    start = time.time()
    my_framework = uncertainty_framework(
        dataset=dataset, 
        T=coalitions, n_random=n_random, n_active=n_active,  v_T=v_values, testset=D_test, 
        kernel=kernel, args=args, embding_func= embding_func,
        aggregate_model = aggregate_model, 
        update_interval = update_interval,
    )

    
    Shapley, variance, v_all_sorted =  my_framework.compute_uncertainty()
    end  = time.time()
    print("Finish in {}", end-start)
    rs =  {'Shapley': Shapley, 'variance': variance, 'v': v_all_sorted, 'time': end-start}
    pd.to_pickle(rs, output_path)
    pd.to_pickle(my_framework, f_model_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data")
    # Training parameters
    parser.add_argument("--device", type=int, default=0,
                        help="Computing device.")
    # embeding function - only true or false
    parser.add_argument('--embd', '--verbose',
                        action='store_true') 
    # our kernel parameters
    parser.add_argument("--device_ids", nargs='+', type=int, default=[]) 
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=3)
    parser.add_argument("--n_projections", type=int, default=100)
    parser.add_argument("--output_dim", type=int, default=2) #0, x only, -1 : 1 hot, -2: y only
    parser.add_argument("--update_interval", type=int, default=0)

    # model aggregate
    parser.add_argument("--model_aggregate", type=str, default="Net",
                        help = "Model for aggregate all data",
                        choices = ["Net", "MNIST_CNN", "ResNet_18_Classifier", "CNNRegressor", "MLPRegressor"])
    #  
    # GP kernel
    parser.add_argument("--training_iteration", default=100, type=int,
                        help="Number of training interations.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--kernel", type=str, default="Exponential_SW_Kernel",
                        choices=["My_OTDD_SW_Kernel", "Exponential_SW_Kernel", "base"], help="Type of Kernel")
    parser.add_argument("--noise", type=float, default=0.1, 
                       help="Noise")
    parser.add_argument("--lengthscale", type=float, default=2, 
                       help="Lengthscale")
    parser.add_argument("--n_random", type=int, default=0, help="Number of random coalitions will be selected for training")
    parser.add_argument("--n_active", type=int, default=0, help="Number of active coalitions will be selected for training")
    parser.add_argument("--seed", type=int, default=24, help="Random seed.")

    parser.add_argument("--out_dir", type=str, required=True,
                        help="Path to output")

    parser.add_argument("--load_data", type=str, default="dsconcat")
    parser.add_argument("--eta", type=float, default=0.5) 
    
    args = parser.parse_args()

    __main__(args)   