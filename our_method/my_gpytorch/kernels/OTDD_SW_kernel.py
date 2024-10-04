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
from ..utils import utils
from .kernel import Kernel 

#
import ot
import itertools
from functools import partial
from ot import wasserstein_1d
#
import ot
import otdd
import numpy as np
from otdd.pytorch.distance import DatasetDistance
from otdd.pytorch.utils import process_device_arg
from tqdm.autonotebook import tqdm
from ..dataset import *
#
import sys
import logging
logger = logging.getLogger(__name__)
def pwdist_exact_SW(X1, Y1, X2=None, Y2=None, projs = None, symmetric=False, loss='sinkhorn',
                 cost_function='euclidean', p=2, debias=True, entreg=1e-1, device='cpu'):

    """ Efficient computation of pairwise label-to-label Wasserstein distances
    between multiple distributions, without using Gaussian assumption.

    Args:
        X1,X2 (tensor): n x d matrix with features
        Y1,Y2 (tensor): labels corresponding to samples
        symmetric (bool): whether X1/Y1 and X2/Y2 are to be treated as the same dataset
        cost_function (callable/string): the 'ground metric' between features to
            be used in optimal transport problem. If callable, should take follow
            the convection of the cost argument in geomloss.SamplesLoss
        p (int): power of the cost (i.e. order of p-Wasserstein distance). Ignored
            if cost_function is a callable.
        debias (bool): Only relevant for Sinkhorn. If true, uses debiased sinkhorn
            divergence.

    """
    device = process_device_arg(device)
    if X2 is None:
        symmetric = True
        X2, Y2 = X1, Y1
        
    c1 = torch.unique(Y1)
    c2 = torch.unique(Y2)
    n1, n2 = len(c1), len(c2)

    if symmetric:
        ## If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        ## If tasks are assymetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))


    logger.info('Computing label-to-label (exact)')
    pbar = tqdm(pairs, leave=False)
    pbar.set_description('Computing label-to-label distances')
    D = torch.zeros((n1, n2), device = device, dtype=X1.dtype)
    for i, j in pbar:
        try:
            D[i, j] = ot.sliced.sliced_wasserstein_distance(X1[Y1==c1[i]].to(device), X2[Y2==c2[j]].to(device), projections=projs)
        except:
            print("This is awkward. Distance computation failed. Geomloss is hard to debug" \
                  "But here's a few things that might be happening: "\
                  " 1. Too many samples with this label, causing memory issues" \
                  " 2. Datatype errors, e.g., if the two datasets have different type")
            sys.exit('Distance computation failed. Aborting.')
        if symmetric:
            D[j, i] = D[i, j]
    return D
#
class my_OTDD_SW(DatasetDistance):
    def __init__(self, D1=None, D2=None,
                 feature_cost='euclidean',
                 ## Inner OT (label to label) problem arguments
                 inner_ot_loss='sinkhorn',
                 inner_ot_debiased=False,
                 inner_ot_p=2,
                 inner_ot_entreg=0.1,
                 ## Gaussian Approximation Args
                 nworkers_dists=0,
                 eigen_correction=False,
                 device='cpu',
                 precision='single',
                 verbose=1, projs = None, *args, **kwargs):
        super(my_OTDD_SW, self).__init__(D1,D2, feature_cost=feature_cost, 
                                         inner_ot_loss='sinkhorn',
                 inner_ot_debiased=inner_ot_debiased,
                 inner_ot_p=inner_ot_p,
                 inner_ot_entreg=inner_ot_entreg,
                    device=device, precision=precision, verbose=verbose)
        # self._init_data(self.D1, self.D2)
        self.projs=projs


    def _get_label_distances(self):
        """ Precompute label-to-label distances.

        Returns tensor of size nclasses_1 x nclasses_2.

        Useful when computing multiple distances on same pair of datasets
        e.g. between subsets of each datasets. Will store them in memory.

        Only useful if method=='precomputed_labeldist', for now.

        Note that _get_label_stats not called for inner_ot_method = `exact`,
        since exact computation does not use Gaussian approximation, so means
        and covariances are not needed.

        Returns:
            label_distances (torch.tensor): tensor of size (C1, C2) with pairwise
                label-to-label distances across the two datasets.

        """
        ## Check if already computed
        if not self.label_distances is None:
            return self.label_distances


        if (self.X1 is None) or (self.X2 is None):
            self._load_datasets(maxsamples=None)  # for now, will use *all* data, to be equiv  to Gaussian

        pwdist = partial(pwdist_exact_SW,
                         projs=self.projs,
                         symmetric=self.symmetric_tasks,
                         p = self.inner_ot_p,
                         loss = self.inner_ot_loss,
                         debias=self.inner_ot_debiased,
                         entreg = self.inner_ot_entreg,
                         cost_function = self.feature_cost,
                         device=self.device)

        

        if self.debiased_loss and not self.symmetric_tasks:
            ## Then we also need within-collection label distances
            if self._pwlabel_stats_1 is None:
                logger.info('Pre-computing pairwise label Slice Wasserstein distances D1 <-> D1...')
                DYY1 = pwdist(self.X1, self.Y1)
            else:
                DYY1 = self._pwlabel_stats_1['dlabs']

            if self._pwlabel_stats_2 is None:
                logger.info('Pre-computing pairwise label Slice Wasserstein distances D2 <-> D2...')
                DYY2 = pwdist(self.X2, self.Y2)
            else:
                logger.info('Found pre-existing D2 label-label stats, will not recompute')
                YY2 = self._pwlabel_stats_2['dlabs']
        else:
            sqrtΣ1, sqrtΣ2 = None, None  # Will have to compute during cross
            DYY1 = DYY2 = None
            DYY1_means = DYY2_means = None

        ## Compute Cross-Distances
        logger.info('Pre-computing pairwise label Wasserstein distances D1 <-> D2...')
        DYY12 = pwdist(self.X1,self.Y1,self.X2, self.Y2)
        DYY12_means = None


        if self.debiased_loss and self.symmetric_tasks:
            ## In this case we can reuse DXY to get DYY1 and DYY
            DYY1, DYY2 = DYY12, DYY12

        if self.debiased_loss:
            D = torch.cat([torch.cat([DYY1, DYY12], 1),
                           torch.cat([DYY12.t(), DYY2], 1)], 0)
        else:
            D = DYY12

        ## Collect and save
        self.label_distances  = D
        self._pwlabel_stats_1 = {'dlabs':DYY1}#
        self._pwlabel_stats_2 = {'dlabs':DYY2}#, 'dmeans':DYY2_means, 'sqrtΣ':sqrtΣ2}

        return self.label_distances

#
def compute_2nint(ic):
    return int(sum([i2*2**i1 for i1,i2 in enumerate(ic)]))
    
def otdd_SW(x1, x2, dataset, projs, p=1, 
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
            dist = my_OTDD_SW(d1, d2,device=device, projs=projs,) 
            rs_dist[ii1, ii2] = dist.distance(maxsamples = 1000)
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

class My_OTDD_SW_Kernel(Kernel):
    """
    """    
    has_lengthscale = True
    def __init__(self, dataset, args={}, n_project=50, square_dist=False, projs=None):
        super(My_OTDD_SW_Kernel, self).__init__(active_dimes=None)
        self.dataset = dataset
        # self.coalitions = coalitions
        self.args = args
        self.dict_sym = {}
        self.SWDD = False
        self.square_dist = square_dist
        device = vars(args).get("device", "cuda:0")
        if projs is not None:
            self.projs = projs.to(device)
        else:
            nfts = dataset[0].data.shape[1] 
            self.projs = torch.tensor(ot.sliced.get_random_projections(nfts, n_project)).float().to(device)
               
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
        p = vars(self.args).get("p", 1)
        device = vars(self.args).get("device", "cuda:0")
        rs = otdd_SW(x1, x2, self.dataset, self.projs, p, self.dict_sym, device)
        if diag:
            rs = torch.diagonal(rs)
        if square_dist:
            rs = torch.square(rs)
        return rs
