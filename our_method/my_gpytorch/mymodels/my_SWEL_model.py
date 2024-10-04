#!/usr/bin/env python3

import warnings
from copy import deepcopy
import argparse
import torch

from .. import settings
from ..distributions import MultitaskMultivariateNormal, MultivariateNormal
from ..likelihoods import _GaussianLikelihoodBase, GaussianLikelihood
from ..utils.generic import length_safe_zip
from ..utils.warnings import GPInputWarning
from ..models.exact_prediction_strategies import prediction_strategy
from ..models import GP
#
#
# from ..kernels import Kernel
# from gpytorch.likelihoods import GaussianLikelihood

from ..models import ApproximateGP
from ..variational import CholeskyVariationalDistribution, VariationalStrategy
from ..mlls.marginal_log_likelihood import MarginalLogLikelihood
# from gpytorch.distributions import MultivariateNormal
from ..means import ConstantMean
from ..distributions import Distribution
#
from dataclasses import dataclass, field
from typing import Optional
#
from torch import Tensor
from torch import FloatTensor
from torch import LongTensor
from torch import IntTensor
from torch import BoolTensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
#
from sklearn.metrics import classification_report, confusion_matrix, auc, accuracy_score
from sklearn.preprocessing import LabelEncoder
# from xgboost import XGBClassifier
#
from torch import FloatTensor
from torch import LongTensor
from torch import IntTensor
from torch import BoolTensor
from ..means import ConstantMean
from ..models import ExactGP
from ..kernels import Kernel, ScaleKernel, RBFKernel, Exponential_SW_Kernel
from ..utils import utils
from ..mlls import ExactMarginalLogLikelihood
from ..label_distance import *
from..constraints import constraints
from ..priors import NormalPrior,GammaPrior
from ..metrics import mean_squared_error, mean_standardized_log_loss,negative_log_predictive_density
from ..dataset import *
import numpy as np
def MBS_mapping_label(dataset, output_dim=2):
    """ 
    if output dim = -1: y , output dim = -2: one hot
    """
    if output_dim == -2:
        targets = [i.targets for i in dataset]
        targets = torch.concat(targets).numpy()
        categories, inverse = np.unique(targets, return_inverse=True)
        _,inverse = np.unique(categories, return_inverse=True)
        mapping_one_hot = np.zeros((categories.size, categories.size))
        mapping_one_hot[np.arange(categories.size), inverse] = 1
        return torch.tensor(mapping_one_hot)
    elif output_dim == -1:
        targets = [i.targets for i in dataset]
        targets = torch.concat(targets)
        return torch.unique(targets).reshape(-1,1)
    label_distance = label_DS_sw_distance(dataset)
    embs =  MDS_emb(label_distance, output_dim=output_dim)
    mapping_label = torch.tensor(embs)
    return mapping_label
    
def transfrom_ds_projs(ids, mapping_label, projs):
    return transform_ds(ids, mapping_label) @ projs

def load_data(i1, dataset):
    idx = utils.from_coalition_to_list([i1.cpu().numpy()])[0]
    lds = [dataset[i] for i in idx]
    return torch.concat(lds)


class OT_SWEL_Model(ExactGP):
    def __init__(self, 
                 train_x: FloatTensor, # [[1, 0, 0], [0, 0, 1]] one hot coding of the coalition position
                 train_y: FloatTensor, 
                 kernel: Kernel,
                 dataset: FloatTensor, 
                 likelihood: GaussianLikelihood, 
                 args: argparse.Namespace,
                 embding_func: callable,
                ) -> None:
        super().__init__(train_x, train_y, likelihood)
        # super(OT_ExactGPRegressionScaleModel, self).__init__(train_x, train_y, likelihood)
        self.args = args

        num_output_dim = vars(self.args).get("output_dim", 2)
        n_projections = vars(self.args).get("n_projections", 50)
        nfts = dataset[0].data.shape[1]
        if embding_func is None:
            mapping_label = None
        else:    
            mapping_label = embding_func(dataset, num_output_dim)
            nfts = nfts + mapping_label.shape[1]
        projs = torch.tensor(ot.sliced.get_random_projections(nfts, n_projections))
        self.dataset = [transfrom_ds_projs(i, mapping_label, projs) for i in dataset]
        self.mean_module = ConstantMean() 
        self.n_player = train_x.shape[1]
        self.load_data = args.load_data
        self.covariance_module =  ScaleKernel(kernel(dataset=self.dataset, args=args))
        self.n_each_party = torch.tensor([i.shape[0] for i in self.dataset])
        self.mean_party = torch.tensor([i.mean() for i in self.dataset])
        # self.covariance_module.base_kernel.lengthscale = 5 # set to the optimal one! run first and set
        
    def set_new(self, dataset):
        self.train_ds = self.dataset
        self.dataset = dataset
        self.covariance_module.set_newds(dataset)
        
    def __mean_coalition__(self, ic):
        idx = utils.from_coalition_to_list([ic.cpu().numpy()])[0]
        sum_ = torch.zeros_like(self.mean_party[0])
        n_ = 0
        for i in idx:
            sum_ = sum_ + self.mean_party[i]*self.n_each_party[i]
            n_ += self.n_each_party[i]
        return sum_/n_
        
    def forward(self, x: FloatTensor) -> MultivariateNormal:
        if x.shape[0] > 1:
            x = x.squeeze()
        ll = []
        for icoalition in x:
        #     d1 = load_data(icoalition, self.dataset)
        #     ll.append(d1.mean(dim=0).unsqueeze(0))
            ll.append(self.__mean_coalition__(icoalition).unsqueeze(0))
        new_input_m = torch.cat(ll, dim=0)
        mean = self.mean_module(new_input_m)
        covariance =  self.covariance_module(x)
        return MultivariateNormal(mean, covariance)
       
def evaluation(pred, y_true):
    mse = mean_squared_error(pred, y_true)
    msll = mean_standardized_log_loss(pred, y_true)
    nlpd = negative_log_predictive_density(pred, y_true)
    return mse, msll, nlpd