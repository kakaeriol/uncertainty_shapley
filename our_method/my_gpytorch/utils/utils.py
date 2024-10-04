import otdd
from otdd.pytorch.distance import DatasetDistance
from sklearn import preprocessing
from itertools import product
from itertools import product
from itertools import combinations
import math
#
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import ot
def mse(y_true, y_pred):
    mse = ((y_pred-y_true)**2).mean()
    return mse


def msll(y_true, y_pred, y_std):
    first_term = 0.5 * np.log(2 * np.pi * y_std**2)
    second_term = ((y_true - y_pred)**2)/(2 * y_std**2)
    
    return np.mean(first_term + second_term)
    
def power_set(List):
    PS = [list(j) for i in range(len(List)) for j in combinations(List, i+1)]
    return PS
def possible_coalitions(n_parties):
    # return np.array(list(product(*([[0,1]]*n_parties))))[1:, :]
    tempList = list([i for i in range(n_parties)])
    coalitions = power_set(tempList)
    return from_list_to_coalitions(coalitions, n_parties)
def from_coalition_to_list(coalitions):
    ll = []
    for irow in coalitions:
        ll.append(np.where(irow)[0].tolist())
    return ll

def from_list_to_coalitions(ll, n_parties):
    n_list = sum((isinstance(i, list) or isinstance(i, (np.ndarray, np.generic) )) for i in ll)
    if n_list == 0: #single array
        arr = np.zeros((1, n_parties))
        for i in ll:
            arr[0, i] = 1
    else:
        arr = np.zeros((n_list, n_parties))
        for ii, ilist in enumerate(ll):
            for i in ilist:
                arr[ii, i] = 1
    return arr

    
class My_Single_Dataset(Dataset):
    """
    This is class to measuring
    """
    def __init__(self,x,y):
        self.data = x
        self.clf = preprocessing.LabelEncoder()
        self.targets = torch.tensor(self.clf.fit_transform(y), dtype=int)
        self.Y = y.int
    # def __init__(self,x):
    #     self.data = x
    #     self.targets = torch.zeros(x.shape[0],  dtype=int)
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, item):
        return self.data[item],self.targets[item]
    def __delitem__(self, key):
        self.data = np.delete(self.data, key, axis=0)
        self.targets = np.delete(self.targets, key, axis=0)

def otdd_dist(idx_1, idx_2, dataset, coalitions, inner_ot_method="exact", p=2, dict_sym={}, device="cuda:0"):
    """
    x1, x2: list of combination of datasets list
    the second last column: labels (if flag is_have_label = True)
    the last column: indicate which is the dataset

    
    """
    
    new_input1  = torch.squeeze(idx_1)
    new_input2  = torch.squeeze(idx_2)
    if idx_1.shape[0] == 1:
        new_input1 = [new_input1]
    if idx_2.shape[0] == 1:
        new_input2 = [new_input2]
    # print(new_input1, new_input1)
    n1 = len(new_input1)
    n2 = len(new_input2)
    rs_dist = torch.zeros((n1, n2),  dtype=torch.double)
    # print("sym dict", dict_sym)
    for ii1, i1 in enumerate(new_input1):
        for ii2, i2 in enumerate(new_input2):
            #
            idx_check = str(sorted([int(i1.numpy()), int(i2.numpy())]))
            if idx_check in dict_sym.keys():
                rs_dist[ii1, ii2] = dict_sym[idx_check]
                continue
            if i1.numpy() == i2.numpy():
                rs_dist[ii1, ii2] = 0
                continue
            
            icoalition1 = coalitions[i1]
            list_all_c1 = from_coalition_to_list([icoalition1])[0]
            ll_x1 = []
            ll_x2 = []
            si1 = dataset[:, -1] == -1
            for tt1 in list_all_c1:
                si1 = si1 | (dataset[:, -1] == tt1)
            X1 = dataset[si1][:, :-2]
            Y1 = dataset[si1][:, -2].int()

            icoalition2 = coalitions[i2]
            list_all_c2 = from_coalition_to_list([icoalition2])[0]
            si2 = dataset[:, -1] == -1
            for tt2 in list_all_c2:
                si2 = si2 | (dataset[:, -1] == tt2)
    
            X2 = dataset[si2][:, :-2]
            Y2 = dataset[si2][:, -2].int()
            d1 = My_Single_Dataset(X1, Y1)
            d2 = My_Single_Dataset(X2, Y2)

            dist = DatasetDistance(d1, d2,
                   # inner_ot_method = 'gaussian_approx',
                   inner_ot_method = inner_ot_method,
                   inner_ot_debiased=True,
                   inner_ot_p = p,
                   debiased_loss = True,
                   p = p, entreg = 1e-1,
                   min_labelcount=1,
                   device=device) # error ot.gpu -- need to check again #cuda:0#device='cpu')
            rs_dist[ii1, ii2] = dist.distance(maxsamples = 1000)
            dict_sym[idx_check] = rs_dist[ii1, ii2]
    return rs_dist

def compute_2nint(ic):
    return int(sum([i2*2**i1 for i1,i2 in enumerate(ic)]))
def otdd_dist_1hot(x1, x2, dataset, inner_ot_method="gaussian_approx", p=1, dict_sym={}, device="cuda:0", lengthscale=None):
    """
    x1, x2: list of combination of datasets list
    the second last column: labels (if flag is_have_label = True)
    the last column: indicate which is the dataset

    
    """
    new_input1  = torch.squeeze(x1) # check if we need squeeze
    new_input2  = torch.squeeze(x2)


    # print(new_input1, new_input1)
    n1 = len(new_input1)
    n2 = len(new_input2)
    rs_dist = torch.zeros((n1, n2),  dtype=torch.double)
    # print("sym dict", dict_sym)
    for ii1, i1 in enumerate(new_input1):
        for ii2, i2 in enumerate(new_input2):
            # print(i1, i2)
            #
            idx_check = str(sorted([compute_2nint(i1.numpy()), compute_2nint(i2.numpy())]))
            if idx_check in dict_sym.keys():
                rs_dist[ii1, ii2] = dict_sym[idx_check]
                continue
            if torch.equal(i1, i2):
                rs_dist[ii1, ii2] = 0
                continue
            
            list_all_c1 = from_coalition_to_list([i1.numpy()])[0]
            ll_x1 = []
            ll_x2 = []
            si1 = dataset[:, -1] == -1 # set all is false
            for tt1 in list_all_c1:
                si1 = si1 | (dataset[:, -1] == tt1)
            X1 = dataset[si1][:, :-2]
            Y1 = dataset[si1][:, -2].int()

            list_all_c2 = from_coalition_to_list([i2.numpy()])[0]
            si2 = dataset[:, -1] == -1
            for tt2 in list_all_c2:
                si2 = si2 | (dataset[:, -1] == tt2)
    
            X2 = dataset[si2][:, :-2]
            Y2 = dataset[si2][:, -2].int()
            d1 = My_Single_Dataset(X1, Y1)
            d2 = My_Single_Dataset(X2, Y2)
            # print(X1.shape, X2.shape, Y1.shape, Y2.shape)

            dist = DatasetDistance(d1, d2,
                   # inner_ot_method = 'gaussian_approx',
                   inner_ot_method = inner_ot_method,
                #    inner_ot_debiased=True,
                   inner_ot_p = p,
                   debiased_loss = True,
                   p = p, entreg = 1e-1,
                   min_labelcount=1,
                   device=device) # error ot.gpu -- need to check again #cuda:0#device='cpu')
            rs_dist[ii1, ii2] = dist.distance(maxsamples = 1000)
            dict_sym[idx_check] = rs_dist[ii1, ii2]
            # print('dis', rs_dist[ii1, ii2])
    return rs_dist


def slice_wasserstein(x1, x2, dataset, dict_sym={}, device="cuda:0"):
    """
    x1, x2: list of combination of datasets list
    the second last column: labels (if flag is_have_label = True)
    the last column: indicate which is the dataset
    """
    new_input1  = torch.squeeze(x1) # check if we need squeeze
    new_input2  = torch.squeeze(x2)
    # print("x1", x1)
    # print("x2", x2)
    # print(new_input1, new_input1)
    n1 = len(new_input1)
    n2 = len(new_input2)
    rs_dist = torch.zeros((n1, n2),  dtype=torch.double)
    # print("sym dict", dict_sym)
    for ii1, i1 in enumerate(new_input1):
        for ii2, i2 in enumerate(new_input2):
            #
            idx_check = str(sorted([compute_2nint(i1.numpy()), compute_2nint(i2.numpy())]))
            if idx_check in dict_sym.keys():
                rs_dist[ii1, ii2] = dict_sym[idx_check]
                continue
            if torch.equal(i1, i2):
                rs_dist[ii1, ii2] = 0
                continue
            
            list_all_c1 = from_coalition_to_list([i1.numpy()])[0]
            ll_x1 = []
            ll_x2 = []
            si1 = dataset[:, -1] == -1 # set all is false
            for tt1 in list_all_c1:
                si1 = si1 | (dataset[:, -1] == tt1)
            X1 = dataset[si1][:, :-1] # including the class

            list_all_c2 = from_coalition_to_list([i2.numpy()])[0]
            si2 = dataset[:, -1] == -1
            for tt2 in list_all_c2:
                si2 = si2 | (dataset[:, -1] == tt2)
    
            X2 = dataset[si2][:, :-1] # including the class
            rs_dist[ii1, ii2]  = ot.sliced_wasserstein_distance(X1.to(device), X2.to(device))
            dict_sym[idx_check] = rs_dist[ii1, ii2]
    return rs_dist
