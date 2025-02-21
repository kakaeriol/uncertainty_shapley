import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Union
from torch import FloatTensor
import argparse
from my_gpytorch import utils
from collections import Counter
from my_gpytorch import utils
from my_gpytorch.kernels import Kernel, ScaleKernel, my_SW_kernel, my_OTDD_OU_kernel, RBFKernel
from my_gpytorch.mymodels import OT_ExactGPRegressionScaleModel, Base_GPRegressionModel, ExactGPScaleRegression_
from copy import deepcopy,copy
# for the model prediction compare
# from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, auc, accuracy_score
from CNN_model import LinearRegression, CNN
import torch
import torch.nn as nn
np.random.seed(12)
torch.manual_seed(12)
import pdb
import math
import torch.nn.functional as F
import torch.optim as optim
from math import comb
from net import *
from my_gpytorch.dataset import *
import torch
from torch import nn
from my_gpytorch.dataset import *
from torch.utils.data import Dataset, DataLoader
import random
def compute_weight(x):
    ll = []
    n = len(x)
    size_ = sum(x)
    for i in range(n):
        size = size_ - x[i]
        w = 1/(n*comb(n-1, int(size)))
        if x[i] == 0:
            ll.append(-1*w)
        else:
            ll.append(w)
    return pd.Series(ll)
      
@dataclass    
class NN_framework(object):
    dataset: FloatTensor
    T: FloatTensor
    n_random: int = 0
    n_active: int = 0 
    v_T: FloatTensor = None
    testset: FloatTensor = None
    model: nn.Module = None
    args: argparse.Namespace = None
    given_index_random: np.ndarray = None
    aggregate_model: callable = None
    
    def __post_init__(self):
        T = self.T
        self.item_train = {}
        _set_random_seed(self.args.seed)
        self.n_predict = len(self.T) - self.n_random - self.n_active
        idx = range(len(self.T))
        self.weight = pd.DataFrame(self.T.numpy()).apply(lambda x: compute_weight(x), axis = 1)
        self.n_add = self.n_random + self.n_active  # there is no active in neural network
        self.indx_rd = np.random.choice(idx, self.n_add, replace=False)
        print(self.n_add, len(self.T))
        if self.given_index_random is None:
            self.indx_rd = np.random.choice(idx, self.n_add, replace=False)    
        else:
            self.indx_rd = self.given_index_random 
        self.add_train_item = list(self.indx_rd)
        X_train = torch.tensor(self.T[self.indx_rd])
        y_train = torch.tensor(self._get_vT(self.indx_rd))
        self.my_model = self.model(T.shape[1])
        trainDs = My_Single_Dataset(X_train, y_train)
        trainDl = DataLoader(trainDs, batch_size=64, shuffle=True)
        self.my_model.train_model(trainDl, device = self.args.device)

    def _get_vT(self, idx):
        def ultility_compute(ic, item_train = {},seed=3):
            if ic in item_train.keys():
                return item_train[ic]
            c = self.T[ic]
            _set_random_seed(seed)
            list_parties = utils.from_coalition_to_list([c])[0]
            ds = load_dataset(list_parties, self.dataset, "dlconcat")
            n_label = ds.dataset.num_class
            if (n_label == 1):
                targets  = self.testset.dataset.targets
                rs = (ds.dataset.targets[0] == targets).sum().item()/(len(targets))
                rs = torch.tensor(rs)
            else:
                aggregate_model = self._init_aggregate_model()
                acc, loss = aggregate_model.train_model(
                    ds, val_loader=self.testset, num_epochs=100, learning_rate=0.001, device = self.args.device)
                rs = acc
                if acc is None: # case regression 
                    rs = loss
                if isinstance(rs, torch.FloatTensor):
                    rs = rs.cpu()
            item_train.update({ic: rs})
            return rs
            
        if self.v_T is None:
            rs = [ultility_compute(ii, self.item_train, self.args.seed) for ii in idx]
            return torch.tensor(rs) 
        else:
            return self.v_T[idx]

    def _init_aggregate_model(self):
        if self.aggregate_model is None:
            self.aggregate_model = Net
        if self.aggregate_model is CNNRegressor:
            num_class = 1
        elif self.aggregate_model is MLPRegressor:
            num_class = 1
        else:
            num_class = self.testset.dataset.num_class

        if self.aggregate_model is ResNet_18_Classifier:
            aggregate_model = self.aggregate_model(num_class) # only use for cifar
        else:
            aggregate_model = self.aggregate_model(self.testset.dataset.data.shape[1], num_class)
        return aggregate_model

    def compute_uncertainty(self, method="update_train"):
        """There is three way to train: 'fantasy' / fast, build in from gpytorch, 
            'update_train': update data, and  'scratch': retrain from scratch,
            default is fantasy
            """
        T = self.T
        idx = range(len(T))
        # -- if n_random = n_active = 0
        if (self.n_random == 0) and (self.n_active == 0):
            v_values = self._get_vT(idx)
            return Shapley_value(v_values, self.T.shape[1]), v_values
        remaining = [i for i in idx if i not in self.indx_rd]

        X_predict = T[remaining].to(self.args.device).float()
        v_pred = self.my_model(X_predict).cpu().detach().numpy()
        #
        add_train_item = self.indx_rd
        y_train_add_list = self._get_vT(add_train_item).cpu().detach().numpy()
        w_a = self.weight.loc[add_train_item,:].values
        w_b = self.weight.loc[remaining,:].values
        Shapley = w_a.T @ np.array(y_train_add_list) + w_b.T @ v_pred
        v_all = np.concatenate([np.array(y_train_add_list).reshape(len(y_train_add_list),1), v_pred],axis=0)
        new_idx= np.concatenate([add_train_item, remaining])
        return Shapley, v_all[np.argsort(new_idx)]
#-------
def _set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def list_A_wth_B(A, B):
    """ """
    return list(set(A) - set(B))

def rCn(n,r):
    f = math.factorial
    return  (f(r)*f(n-r-1))/f(n)
    
def sublist(lst1, lst2):
    return set(lst1) <= set(lst2)

def possible_combined(idx, n_parties):
    ll = []
    A = utils.possible_coalitions(n_parties)
    for i in utils.from_coalition_to_list(A):
        if sublist(idx, i):
            ll.append(i)
    return ll

def return_index(i_coalition, all_coaltion):
    return np.where(np.abs(all_coaltion - i_coalition).sum(axis = 1) == 0)[0][0]
#----------------------

def i_shapley_values(v, n_parties, idx ):
    """
    Check again shapley value here
    """
    A = utils.possible_coalitions(n_parties)
    poss = possible_combined(idx, n_parties)
    Sh =  0
    for sub_list in poss:
        idx_include = return_index(utils.from_list_to_coalitions(sub_list, n_parties)[0], A)
        v_incl = v[idx_include]
        exclude_list = list_A_wth_B(sub_list, idx)
        if len(exclude_list) == 0:
            v_excl = 0
            idx_exclude = -1
        else:
            idx_exclude = return_index(utils.from_list_to_coalitions(exclude_list, n_parties)[0], A)
            v_excl = v[idx_exclude]
        k = len(exclude_list)
        i_S = rCn(n_parties, len(exclude_list))*(v_incl - v_excl)
        Sh += i_S
    if Sh < 0:
        Sh = 0
    return Sh

def Shapley_value(v_values, n_parties):
    Actual_shapley = np.zeros(n_parties)
    for i in range(n_parties):
        Actual_shapley[i] = i_shapley_values(v_values, n_parties, [i] )
    return Actual_shapley


