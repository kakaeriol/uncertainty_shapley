#!/usr/bin/env python3

from __future__ import annotations

import warnings
from abc import abstractmethod
from copy import deepcopy
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from linear_operator import to_dense, to_linear_operator
from linear_operator.operators import LinearOperator, ZeroLinearOperator
from torch import Tensor
from torch.nn import ModuleList

from .. import settings
from ..constraints import Interval, Positive
from ..distributions import MultivariateNormal
from ..lazy import LazyEvaluatedKernelTensor
from ..likelihoods import GaussianLikelihood
from ..models import exact_prediction_strategies
from ..module import Module
from ..priors import Prior
from ..utils import *
from ..dataset import *
from ..utils import utils
from .kernel import Kernel 


import ot
from ot import wasserstein_1d


def compute_2nint(ic):
    return int(sum([i2*2**i1 for i1,i2 in enumerate(ic)]))

def SW(x1, x2, dataset, projs, p, dict_sym={}, device="cuda:0", pre_compute=False, load_data="dsconcat"):
    """
    x1, x2: list of combination of datasets list
    the second last column: labels (if flag is_have_label = True)
    the last column: indicate which is the dataset
    """
    # new_input1  = torch.squeeze(x1) # check if we need squeeze
    # new_input2  = torch.squeeze(x2)

    new_input1 = x1
    new_input2 = x2
    n1 = len(new_input1)
    n2 = len(new_input2)
    n_projections = projs.shape[1]   
    rs_dist = torch.zeros((n1, n2),  dtype=torch.double)
    # print(n1,n2)
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
            
            list_all_c1 = utils.from_coalition_to_list([i1.numpy()])[0]
            list_all_c2 = utils.from_coalition_to_list([i2.numpy()])[0]
            d1 = load_dataset(list_all_c1, dataset, load_data)
            d2 = load_dataset(list_all_c2, dataset, load_data)
            if pre_compute:
                projected_emd = wasserstein_1d(d1, d2, p=p)
            else:
                if load_data == 'dsconcat':
                    X1 = torch.column_stack([d1.data, d1.targets]).to(device)
                    X2 = torch.column_stack([d2.data, d2.targets]).to(device)
                    X_1_projections = X1 @ projs
                    X_2_projections = X2 @ projs
                    projected_emd = wasserstein_1d(X_1_projections, X_2_projections, p=p)
                elif load_data == 'dlconcat': # batchsize
                    k = 0
                    for d1data, d1target in d1:
                        X1 = torch.column_stack([d1data, d1target]).to(device)
                        X_1_projections = X1 @ projs
                        for d2data, d2target in d2:
                            X2 = torch.column_stack([d2data, d2target]).to(device) # change it later
                            X_2_projections = X2 @ projs
                            if k == 0:
                                projected_emd = wasserstein_1d(X_1_projections, X_2_projections, p=p)
                            else:
                                k = k + 1
                                projected_emd += wasserstein_1d(X_1_projections, X_2_projections, p=p)
                    projected_emd = projected_emd/k
                    # print(ii1, ii2, projected_emd.shape)
                rs_dist[ii1, ii2] = (torch.sum(projected_emd) / n_projections) ** (1.0 / p)
                dict_sym[idx_check] = rs_dist[ii1, ii2]
    return rs_dist


def SWDD(x1, x2, dataset, projs, p, dict_sym={}, device="cuda:0", pre_compute=False, load_data="dsconcat"):
    """
    x1, x2: list of combination of datasets list
    the second last column: labels (if flag is_have_label = True)
    the last column: indicate which is the dataset
    """
    new_input1  = torch.squeeze(x1) # check if we need squeeze
    new_input2  = torch.squeeze(x2)
    # new_input1  = x1
    # new_input2  = x2

    n1 = len(new_input1)
    n2 = len(new_input2)
    n_projections = projs.shape[1]   
    rs_dist = torch.zeros((n1, n2),  dtype=torch.double)
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
            
            list_all_c1 = utils.from_coalition_to_list([i1.numpy()])[0]
            list_all_c2 = utils.from_coalition_to_list([i2.numpy()])[0]
            d1 = load_dataset(list_all_c1, dataset, load_data)
            d2 = load_dataset(list_all_c2, dataset, load_data)
            if pre_compute:
                X_1_projections = d1.data.to(device)
                X_2_projections = d2.data.to(device)
                for ic1 in d1.targets.unique():
                        for ic2 in d2.targets.unique():
                            iX1 = X_1_projections[d1.targets == ic1]
                            iX2 = X_2_projections[d2.targets == ic2]
                            if k == 0:
                                projected_emd = wasserstein_1d(iX1, iX2, p=p)
                            else:
                                k = k + 1
                                projected_emd += wasserstein_1d(iX1, iX2, p=p)
            else:
                k = 0
                if load_data == 'dsconcat':
                    X1 = d1.data.to(device)
                    X2 = d2.data.to(device) 
                    X_1_projections = X1 @ projs
                    X_2_projections = X2 @ projs
                    for ic1 in d1.targets.unique():
                        for ic2 in d2.targets.unique():
                            iX1 = X_1_projections[d1.targets == ic1]
                            iX2 = X_2_projections[d1.targets == ic2]
                            if k == 0:
                                projected_emd = wasserstein_1d(iX1, iX2, p=p)
                            else:
                                k = k + 1
                                projected_emd += wasserstein_1d(iX1, iX2, p=p)
                # need to write to the batch one
            
            rs_dist[ii1, ii2] = (torch.sum(projected_emd) / n_projections) ** (1.0 / p)
            dict_sym[idx_check] = rs_dist[ii1, ii2]
    return rs_dist

def postprocess_rbf(dist_mat):
    return dist_mat.div_(-1).exp_()
    
class SW_Covariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, D1, D2, lengthscale, dfunction):
        if any(ctx.needs_input_grad[:2]):
            raise RuntimeError("RBFCovariance cannot compute gradients with " "respect to x1 and x2")
        if lengthscale.size(-1) > 1:
            raise ValueError("RBFCovariance cannot handle multiple lengthscales")
        needs_grad = any(ctx.needs_input_grad)
        
        unitless_sq_dist = dfunction(D1, D2)
        unitless_sq_dist = unitless_sq_dist.div(lengthscale)
        # clone because inplace operations will mess with what's saved for backward
        unitless_sq_dist_ = unitless_sq_dist.clone() if needs_grad else unitless_sq_dist
        covar_mat = unitless_sq_dist_.div_(-1.0).exp_()
        if needs_grad:
            d_output_d_input = unitless_sq_dist.mul_(covar_mat).div_(lengthscale)
            ctx.save_for_backward(d_output_d_input)
        return covar_mat

    @staticmethod
    def backward(ctx, grad_output):
        d_output_d_input = ctx.saved_tensors[0]
        lengthscale_grad = grad_output * d_output_d_input 
        return None, None, lengthscale_grad, None

class my_SW_kernel(Kernel):
    """
    """    
    has_lengthscale = True
    def __init__(self, dataset, args={}, n_project=50, square_dist=False, pre_compute=False, projs=None, SWDD=False):
        super(my_SW_kernel, self).__init__(active_dimes=None)
        self.dataset = dataset
        # self.coalitions = coalitions
        self.args = args
        self.dict_sym = {}
        self.SWDD = False
        device = vars(args).get("device", "cuda:0")
        if projs is not None:
            self.projs = projs.to(device)
        else:
            nfts = dataset[0].data.shape[1] 
            if SWDD:
                self.projs = torch.tensor(ot.sliced.get_random_projections(nfts, n_project)).to(device)
                self.SWDD = True
            else:
                self.projs = torch.tensor(ot.sliced.get_random_projections(nfts+1, n_project)).to(device)

        self.pre_compute = pre_compute
        self.square_dist = square_dist
        if pre_compute:
            ll = []
            for ds in dataset:
                for ic in ds.target.unique():
                    ids = ds.data.to(device) @ self.projs
                    newds = My_Single_Dataset(ids, ds.data.target)
                    ll.append(newds)
            self.dataset = ll


    def set_newds(self, dataset):
        self.train_ds = self.dataset
        self.dataset = dataset
    def forward(self, D1, D2, diag=False, last_dim_is_batch=False, **params):
        if (
            D1.requires_grad
            or D2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            # or trace_mode.on()
        ):
            return postprocess_rbf(self.covar_dist(D1, D2, diag=diag, square_dist=self.square_dist, **params).div(self.lengthscale))
            
        return SW_Covariance.apply(D1, D2, self.lengthscale, lambda D1, D2: self.covar_dist(D1, D2, diag=False, square_dist=self.square_dist,**params),)
    def covar_dist(self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        square_dist: bool = False,
        **params,
    ) -> Tensor:
        inner_ot_method = vars(self.args).get("inner_ot_method", "gaussian_approx")
        # inner_ot_method = vars(self.args).get("inner_ot_method", "exact")
        p = vars(self.args).get("p", 1)
        device = vars(self.args).get("device", "cuda:0")
        # print(device)
        if self.SWDD:
            rs = SWDD(x1, x2, self.dataset, self.projs, p, self.dict_sym, device,self.pre_compute)
        else:
            rs = SW(x1, x2, self.dataset, self.projs, p, self.dict_sym, device,self.pre_compute)
        if diag:
            rs = torch.diagonal(rs)
        if square_dist:
            rs = torch.square(rs)
        return rs