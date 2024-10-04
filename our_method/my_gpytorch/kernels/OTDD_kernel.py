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
# from ..utils import otdd_dist, otdd_dist_1hot
from .kernel import Kernel 
from ..utils import utils

from ..dataset import *
from otdd.pytorch.distance import DatasetDistance

def compute_2nint(ic):
    return int(sum([i2*2**i1 for i1,i2 in enumerate(ic)]))
    
def Bruser_otdd(x1, x2, dataset, inner_ot_method="gaussian_approx", p=1, 
                   dict_sym={}, device="cuda:0", lengthscale=None, load_data="dsconcat"):
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
    rs_dist = torch.zeros((n1, n2),  dtype=torch.double)
    for ii1, i1 in enumerate(new_input1):
        for ii2, i2 in enumerate(new_input2):
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
            dist = DatasetDistance(d1, d2,
                   inner_ot_method = inner_ot_method,
                   inner_ot_p = p,
                   debiased_loss = True,
                   p = p, entreg = 1e-1,
                   min_labelcount=1,
                   pre_computed=True,
                   device=device) 
            rs_dist[ii1, ii2] = dist.distance(maxsamples = 1000)
            dict_sym[idx_check] = rs_dist[ii1, ii2]
    return rs_dist 


def postprocess_rbf(dist_mat):
    return (dist_mat).div_(-2).exp_()
    
class my_OTDD_RBFCovariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, D1, D2, lengthscale, dfunction):
        if any(ctx.needs_input_grad[:2]):
            raise RuntimeError("RBFCovariance cannot compute gradients with " "respect to x1 and x2")
        if lengthscale.size(-1) > 1:
            raise ValueError("RBFCovariance cannot handle multiple lengthscales")
        needs_grad = any(ctx.needs_input_grad)
        
        unitless_sq_dist = dfunction(D1, D2)
        unitless_sq_dist = unitless_sq_dist.div(lengthscale*lengthscale)
        unitless_sq_dist_ = unitless_sq_dist.clone() if needs_grad else unitless_sq_dist
        covar_mat = unitless_sq_dist_.div_(-2).exp_()

        if needs_grad:
            d_output_d_input = unitless_sq_dist.mul_(covar_mat).div_(lengthscale)
            ctx.save_for_backward(d_output_d_input)
        return covar_mat

    @staticmethod
    def backward(ctx, grad_output):
        d_output_d_input = ctx.saved_tensors[0]
        lengthscale_grad = grad_output * d_output_d_input
        return None, None, lengthscale_grad, None

class my_OTDD_RBF_kernel(Kernel):
    """
    """    
    has_lengthscale = True
    def __init__(self, dataset, args):
        super(my_OTDD_RBF_kernel, self).__init__(active_dimes=None)
        self.dataset = dataset
        self.args = args
        self.dict_sym = {}

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
           
            return postprocess_rbf(self.covar_dist(D1, D2, square_dist=True, diag=diag, **params).div(lengthscale*lengthscale))

        return my_OTDD_RBFCovariance.apply(D1, D2, self.lengthscale, lambda D1, D2: self.covar_dist(D1, D2, square_dist=True,diag=False, **params),)

    def covar_dist(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        square_dist=False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor:
        inner_ot_method = vars(self.args).get("inner_ot_method", "gaussian_approx") # 
        p = vars(self.args).get("p", 1)
        device = vars(self.args).get("device", "cuda:0")
        #
        rs = Bruser_otdd(x1, x2, self.dataset, inner_ot_method, p, self.dict_sym, device)

        if diag:
            rs = torch.diagonal(rs)
        if square_dist:
            rs = torch.square(rs)
        return rs
####
def postprocess_ou_rbf(dist_mat):
    return dist_mat.div_(-1).exp_()
    
class my_OTDD_OU_Covariance(torch.autograd.Function):
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

class my_OTDD_OU_kernel(Kernel):
    """
    """    
    has_lengthscale = True
    def __init__(self, dataset, args):
        super(my_OTDD_OU_kernel, self).__init__(active_dimes=None)
        self.dataset = dataset
        # self.coalitions = coalitions
        self.args = args
        self.dict_sym = {}

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
            return postprocess_ou_rbf(self.covar_dist(D1, D2, square_dist=False, diag=diag, **params).div(self.lengthscale))
        return my_OTDD_OU_Covariance.apply(D1, D2, self.lengthscale, lambda D1, D2: self.covar_dist(D1, D2, square_dist=False, diag=False, **params),)


    def covar_dist(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        square_dist: bool = False,
        **params,
    ) -> Tensor:
        inner_ot_method = vars(self.args).get("inner_ot_method", "gaussian_approx")
        p = vars(self.args).get("p", 2)
        device = vars(self.args).get("device", "cuda:0")
        rs = Bruser_otdd(x1, x2, self.dataset, inner_ot_method, p, self.dict_sym, device)
        if diag:
            rs = torch.diagonal(rs)
        if square_dist:
            rs = torch.square(rs)
        return rs

