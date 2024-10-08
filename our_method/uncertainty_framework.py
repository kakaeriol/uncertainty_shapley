import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Union
from torch import FloatTensor
import argparse
from my_gpytorch import utils
from collections import Counter
from my_gpytorch import utils
from my_gpytorch.kernels import Kernel, ScaleKernel, my_SW_kernel, my_OTDD_OU_kernel, RBFKernel, Exponential_SW_Kernel
from my_gpytorch.mymodels import OT_ExactGPRegressionScaleModel, Base_GPRegressionModel, ExactGPScaleRegression_
from copy import deepcopy,copy
# for the model prediction compare
# from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, auc, accuracy_score
import torch
import torch.nn as nn

import pdb
import math
import torch.nn.functional as F
import torch.optim as optim
from math import comb
from net import *
from my_gpytorch.dataset import *
from my_gpytorch.mymodels import OT_SWEL_Model
#
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

def flatten_single(idataset):
    if idataset.data.ndim == 3:
        return idataset
    x = idataset.data
    x = x.view(x.size(0), -1) .double()
    labels = idataset.targets
    return My_Single_Dataset(x, labels)

def map3c_1c(dataset):
    return [flatten_single(i) for i in dataset]
    
    
@dataclass    
class uncertainty_framework(object):
    dataset: FloatTensor
    T: FloatTensor
    n_random: int = 0
    n_active: int = 0 
    v_T: FloatTensor = None
    testset: FloatTensor = None
    kernel: Kernel = my_SW_kernel
    args: argparse.Namespace = None
    given_index_random: np.ndarray = None
    embding_func: callable = None
    aggregate_model: callable = None
    update_interval: int = None  
    def __post_init__(self):
        self.n_predict = len(self.T) - self.n_random - self.n_active

        idx = range(len(self.T))
        self.item_train = {}
        # init weight
        self.weight = pd.DataFrame(self.T.numpy()).apply(lambda x: compute_weight(x), axis = 1)
        # init the aggregate model
        
        _set_random_seed(self.args.seed)
        self._select_regression_model()
        print(self)
        if self.n_random > 0:
            if self.given_index_random is None:
                self.indx_rd = np.random.choice(idx, self.n_random, replace=False)

            else:
                self.indx_rd = self.given_index_random 
            self.add_train_item = list(self.indx_rd)
            X_train = self.T[self.indx_rd]
            y_train = self._get_vT(self.indx_rd)
            self.y_train_add = y_train

            self.my_model = ExactGPScaleRegression_(X_train, y_train, self.kernel, 
                                                    map3c_1c(self.dataset),args=self.args,
                                                    regression_model=self.regression_model,
                                                    base = self.base,
                                                    embding_func = self.embding_func
                                                   )
            self.my_model.fit(learning_rate=self.args.learning_rate, 
                              training_iteration=self.args.training_iteration,
                              verbose=False, debug=False)

            print("Finish")
            with torch.no_grad():
                self.sigma2 = self.my_model.likelihood.noise.item()
                self.K = self.my_model.model.covariance_module(X_train).evaluate() + self.sigma2*torch.eye(len(X_train))
                self.K_inv = torch.inverse(self.K)
        else:
            self.add_train_item = []
            self.y_train_add = torch.tensor([])
            self.K = torch.tensor([])
            self.K_inv = torch.tensor([])
            
            
    def _select_regression_model(self):
        self.base = False 
        if self.kernel == RBFKernel:
            self.base = True
            self.regression_model = Base_GPRegressionModel
        elif self.kernel == Exponential_SW_Kernel:
            self.regression_model = OT_SWEL_Model
        else:
            self.regression_model = OT_ExactGPRegressionScaleModel

    def _init_aggregate_model(self):
        if self.aggregate_model is None:
            self.aggregate_model = Net
        if self.aggregate_model is CNNRegressor:
            num_class = 1
        else:
            num_class = self.testset.dataset.num_class

        if self.aggregate_model is ResNet_18_Classifier:
            aggregate_model = self.aggregate_model(num_class) # only use for cifar
        else:
            aggregate_model = self.aggregate_model(self.testset.dataset.data.shape[1], num_class)
        return aggregate_model
            
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
    def select_first_coalition(self, remaining):
        ## Option 1: Random selection
        # C_star = np.random.choice(remaining)
        ## Option 2: get start by the one havign a lot of parties
        T_remaining = self.T.numpy()[remaining]
        sums_remaining = T_remaining.sum(axis=1)
        index_in_remaining = np.argmax(sums_remaining)
        C_star = remaining[index_in_remaining]
        return C_star

    def compute_uncertainty(self):
        T = self.T
        idx = range(len(T))
        # -- if n_random = n_active = 0
        if (self.n_random == 0) and (self.n_active == 0):
            v_values = self._get_vT(idx)
            variance = np.zeros((1, self.T.shape[1]))
            return Shapley_value(v_values, self.T.shape[1]), variance, v_values

        # Initialize variables
        remaining, update_model, add_train_item, y_train_add_list = self._initialize_variables(idx)

        # Determine the GP model update interval
        update_gp_interval = self._determine_update_interval()

        for i in range(self.n_active):
            VR_max = -np.inf
            C_star = None
            K_inv_star = None
            if len(add_train_item) == 0:
                # Select the first coalition without prior variance
                C_star = self.select_first_coalition(remaining)
                if C_star is None:
                    break  # No valid candidate found
                add_train_item.append(C_star)
                remaining.remove(C_star)
                y_train_add_list, update_model = self._update_gp_model(add_train_item, update_model)
                self._update_kernel_matrices(update_model, add_train_item)
                continue
            # Select the candidate with maximum variance reduction
            C_star, K_inv_star, K_star, VR_max = self._greedy_active_selection(remaining, add_train_item, VR_max)
            if C_star is None:
                break  # No valid candidate found

            add_train_item.append(C_star)
            y_add = self._get_vT([C_star]).item()
            y_train_add_list.append(y_add)
            remaining.remove(C_star)
            # Update GP model if needed
            update_gp_this_iter = False
            if (((update_gp_interval is not None) and ((i + 1) % update_gp_interval == 0)) or (i == self.n_active-1)):
                update_gp_this_iter = True
            if update_gp_this_iter:        
                y_train_add_list, update_model = self._update_gp_model(add_train_item, update_model)
                self._update_kernel_matrices(update_model, add_train_item)
            else:
                self.K_inv = K_inv_star.float()
                self.K = K_star
        self.add_train_item = add_train_item
        self.y_train_add = torch.tensor(y_train_add_list)
        # update when have all of data
       
        Shapley, variance, v_all = self._make_predictions()
        return Shapley, variance, v_all


    def _update_model_uncertainty(self, n_random, n_active):
        """ Use only need to update the framework, after calling compute_uncertainty"""
        T = self.T
        idx = range(len(T))
        # Initialize variables
        remaining, update_model, add_train_item, y_train_add_list = self._initialize_variables(idx)

        # Determine the GP model update interval
        update_gp_interval = self._determine_update_interval()

        if n_random > 0:
            add_new = np.random.choice(remaining, self.n_random, replace=False)
            add_train_item.append(add_new)
            for ic in add_new:
                remaining.remove(ic)
            y_train_add_list, update_model = self._update_gp_model(add_train_item, update_model)
        
        for i in range(n_active):
            VR_max = 0 
            C_star = None
            K_inv_star = None
            C_star, K_inv_star, K_star, VR_max = self._greedy_active_selection(remaining, add_train_item, VR_max)
            if C_star is None:
                break  # No valid candidate found

            add_train_item.append(C_star)
            y_add = self._get_vT([C_star]).item()
            y_train_add_list.append(y_add)
            remaining.remove(C_star)
            # Update GP model if needed
            update_gp_this_iter = False
            if (((update_gp_interval is not None) and ((i + 1) % update_gp_interval == 0)) or (i == n_active-1)):
                update_gp_this_iter = True
            if update_gp_this_iter:        
                y_train_add_list, update_model = self._update_gp_model(add_train_item, update_model)
                self._update_kernel_matrices(update_model, add_train_item)
            else:
                self.K_inv = K_inv_star.float()
                self.K = K_star
    
        self.add_train_item = add_train_item
        self.y_train_add = torch.tensor(y_train_add_list)
        Shapley, variance, v_all = self._make_predictions()
        return Shapley, variance, v_all


    def _initialize_variables(self, idx):
        if len(self.add_train_item) > 0:
            remaining = [i for i in idx if i not in self.add_train_item]
            update_model = copy(self.my_model)
            add_train_item = self.add_train_item.copy()
            y_train_add_list = list(self.y_train_add.detach().numpy())
        else:
            remaining = list(idx)
            update_model = None  # Will be initialized when updating GP model
            add_train_item = []
            y_train_add_list = []
        return remaining, update_model, add_train_item, y_train_add_list



    def _determine_update_interval(self):
        if self.update_interval is not None and self.update_interval > 0:
            return self.update_interval
        else:
            return None  # No need to update

    def _update_gp_model(self, add_train_item, update_model):
        X_train_add = self.T[add_train_item]
        y_train_add = self._get_vT(add_train_item)
        y_train_add_list = list(y_train_add.detach().numpy())
        if update_model is None:
            X_train = X_train_add
            y_train = y_train_add
            update_model = ExactGPScaleRegression_(X_train, y_train,self.kernel,
                                                   map3c_1c(self.dataset), args=self.args,
                                                   regression_model=self.regression_model,
                                                   base=self.base,
                                                   embding_func=self.embding_func)
        else:
            # Update existing model
            update_model.set_train_data(X_train_add, y_train_add)
        update_model.fit(learning_rate=self.args.learning_rate,
                         training_iteration=50, verbose=False, debug=False)
        return y_train_add_list, update_model

    def _update_kernel_matrices(self, update_model, add_train_item):
        with torch.no_grad():
            self.sigma2 = update_model.likelihood.noise.item()
            X_train_add = self.T[add_train_item]
            self.K = update_model.model.covariance_module(X_train_add).evaluate() + self.sigma2 * torch.eye(len(X_train_add))
            self.K_inv = torch.inverse(self.K).float()
        self.my_model = update_model

    def _greedy_active_selection(self, remaining, add_train_item, VR_max):
        C_star = None
        K_inv_star = None
        K_star = None
        # print(remaining, add_train_item)
        for C_candidate in remaining:
            C_candidate_idx = [C_candidate]
            with torch.no_grad():
                k_vec = self.my_model.model.covariance_module(
                    self.T[C_candidate_idx], self.T[add_train_item]
                ).evaluate().squeeze(0).float()  # Shape: (n,)
                k_CC = self.my_model.model.covariance_module(
                    self.T[C_candidate_idx], self.T[C_candidate_idx]
                ).evaluate().float() + self.sigma2  # Scalar
                
                self.K_inv = self.K_inv.float()
                
                # Compute s
                k_vec = k_vec.unsqueeze(1)  # Shape: (n, 1)
                v = self.K_inv @ k_vec  # Shape: (n, 1)
                s = (k_CC - (k_vec.T @ v)).item()
                if s <= 0:
                    continue  # Skip this candidate due to numerical issues
                
                # Compute S_inv
                S_inv = 1.0 / s
                
                # Incremental update of K_inv_candidate
                K_inv_candidate = torch.zeros(len(add_train_item) + 1, len(add_train_item) + 1)
                K_inv_candidate[:-1, :-1] = self.K_inv + (v @ v.T) * S_inv
                K_inv_candidate[:-1, -1] = -v.squeeze() * S_inv
                K_inv_candidate[-1, :-1] = (-v.squeeze() * S_inv).T
                K_inv_candidate[-1, -1] = S_inv

                # Compute K candidate
                k_vec_squeezed = k_vec.squeeze(1)  # Shape: (n,)
                K_candidate = torch.zeros(len(add_train_item) + 1, len(add_train_item) + 1)
                K_candidate[:-1, :-1] = self.K
                K_candidate[:-1, -1] = k_vec_squeezed
                K_candidate[-1, :-1] = k_vec_squeezed.T
                K_candidate[-1, -1] = k_CC
                
                # Compute w_B_i and K_BA_candidate
                w_B_i = self.weight.iloc[remaining, :].values
                w_B_i = torch.tensor(w_B_i, dtype=torch.float32)
                K_BA_candidate = self.my_model.model.covariance_module(
                    self.T[remaining], self.T[add_train_item + C_candidate_idx]
                ).evaluate().float()
                
                # Compute VR_C
                temp = K_BA_candidate @ K_inv_candidate
                VR_C = (w_B_i.T @ temp @ temp.T @ w_B_i).sum().item()

            # Select the candidate with the maximum variance reduction
            if VR_C > VR_max:
                VR_max = VR_C
                C_star = C_candidate
                K_inv_star = K_inv_candidate
                K_star = K_candidate
        return C_star, K_inv_star, K_star, VR_max

    def _make_predictions(self):
        add_train_item = self.add_train_item
        remaining = [i for i in range(len(self.T)) if i not in add_train_item]
        y_train_add_list = self.y_train_add
        X_predict = self.T[remaining]
        v_pred = self.my_model.predict(X_predict)
        Variance_matrix = v_pred.covariance_matrix.detach().numpy()
        w_a = self.weight.iloc[add_train_item, :].values
        w_b = self.weight.iloc[remaining, :].values
        if v_pred.mean.detach().numpy().ndim == 2:
            Shapley = w_a.T @ np.array(y_train_add_list) + w_b.T @ v_pred.mean.detach().numpy()[0]
            v_all = np.hstack([np.array(y_train_add_list), v_pred.mean.detach().numpy()[0]])
        else:
            Shapley = w_a.T @ np.array(y_train_add_list) + w_b.T @ v_pred.mean.detach().numpy()
            v_all = np.hstack([np.array(y_train_add_list), v_pred.mean.detach().numpy()])

        variance = np.diagonal(w_b.T @ Variance_matrix @ w_b)
        new_idx = np.hstack([add_train_item, remaining])
        v_all_sorted = v_all[np.argsort(new_idx)]
        return Shapley, variance, v_all_sorted

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
        if v_incl < 0:
            v_incl = 0
        if v_excl < 0:
            v_excl = 0
        i_S = rCn(n_parties, len(exclude_list))*(v_incl - v_excl)
        # if i_S < 0:
        #     i_S = 0
        # print(sub_list, exclude_list)
        # print(idx_include, idx_exclude)
        # print(v_incl,v_excl, i_S)
        Sh += i_S
    if Sh < 0:
        Sh = 0
    return Sh

def Shapley_value(v_values, n_parties):
    Actual_shapley = np.zeros(n_parties)
    for i in range(n_parties):
        Actual_shapley[i] = i_shapley_values(v_values, n_parties, [i] )
    return Actual_shapley


