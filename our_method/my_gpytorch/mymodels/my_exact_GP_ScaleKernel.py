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
from ..kernels import Kernel, ScaleKernel, RBFKernel
from ..utils import utils
from ..mlls import ExactMarginalLogLikelihood

from..constraints import constraints
from ..priors import NormalPrior,GammaPrior
from ..metrics import mean_squared_error, mean_standardized_log_loss,negative_log_predictive_density
from ..dataset import *
from .my_SWEL_model import OT_SWEL_Model
# settings.lazily_evaluate_kernels(False)
# class ExactGPRegressionModel(gpytorch.models.ExactGP):
class OT_ExactGPRegressionScaleModel(ExactGP):
    def __init__(self, 
                 train_x: FloatTensor, # [[1, 0, 0], [0, 0, 1]] one hot coding of the coalition position
                 train_y: FloatTensor, 
                 kernel: Kernel,
                 dataset: FloatTensor, 
                 likelihood: GaussianLikelihood, 
                 args: argparse.Namespace,) -> None:
        super(OT_ExactGPRegressionScaleModel, self).__init__(train_x, train_y, likelihood)
        self.dataset = dataset
        self.mean_module = ConstantMean() 
        self.n_player = train_x.shape[1]
        self.load_data = args.load_data
        self.covariance_module =  ScaleKernel(kernel(dataset=self.dataset, args=args)) #my_OTDD_RBF_kernel()
        self.covariance_module.base_kernel.lengthscale = 5 # set to the optimal one! run first and set
        
    def set_new(self, dataset):
        self.train_ds = self.dataset
        self.dataset = dataset
        self.covariance_module.set_newds(dataset)
        
    def forward(self, x: FloatTensor) -> MultivariateNormal:
        x = x.squeeze()
        ll = []
        for icoalition in x:
            ll0 = []
            list_all_c = utils.from_coalition_to_list([icoalition])[0]
            d1 = load_dataset(list_all_c, self.dataset, self.load_data)
            ll.append(d1.data.mean(dim=0).unsqueeze(0))
        new_input_m = torch.cat(ll, dim=0)
        covariance =  self.covariance_module(x)
        return MultivariateNormal(self.mean_module(new_input_m),
                                 covariance)
       
class Base_GPRegressionModel(ExactGP):
    def __init__(self, train_x, train_y, kernel = RBFKernel(), likelihood=GaussianLikelihood()):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covariance_module = ScaleKernel(kernel())
        self.covariance_module.base_kernel.lengthscale = 10
     
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covariance_module(x)
        return MultivariateNormal(mean_x, covar_x)
        
@dataclass()
class ExactGPScaleRegression_(object):
    train_X: FloatTensor
    train_y: FloatTensor
    kernel: Kernel
    dataset: FloatTensor
    lengthscale: FloatTensor = field(init=False, default=None)
    optimizer: Optimizer = field(init=False, default=None)
    marginal_log_likelihood: MarginalLogLikelihood = field(init=False, default=None)
    args: argparse.Namespace
    regression_model : ExactGP = OT_ExactGPRegressionScaleModel
    embding_func: callable = None
    base: bool = False

    def __post_init__(self):
        # self.likelihood = GaussianLikelihood()
        self.likelihood = GaussianLikelihood(
            # noise_constraint=constraints.GreaterThan(1e-3),
            # noise_prior=NormalPrior(0, 1)
        )
        if self.base:
            self.model = self.regression_model(self.train_X, self.train_y, self.kernel)
        elif self.regression_model is OT_SWEL_Model:
            # print("train y", self.train_y.shape)
            self.model = self.regression_model(self.train_X, self.train_y, self.kernel, self.dataset, self.likelihood,  self.args, self.embding_func)
        else:
            self.model = self.regression_model(self.train_X, self.train_y, self.kernel, self.dataset, self.likelihood,  self.args)
        init_noise = self.args.noise
        init_lengthscale = self.args.lengthscale
        hypers = {
            'likelihood.noise_covar.noise': torch.tensor(init_noise),
            'covariance_module.base_kernel.lengthscale': torch.tensor(init_lengthscale),
            'covariance_module.outputscale': torch.tensor(1),
        }
        
        self.model.initialize(**hypers)
        print(
            self.model.likelihood.noise_covar.noise.item(),
            self.model.likelihood.noise_covar.raw_noise_constraint,
            self.model.covariance_module.base_kernel.lengthscale.item(),
            self.model.covariance_module.outputscale.item(),
            self.model.covariance_module.raw_outputscale_constraint,
        )

        # enter train mode
        self.loss = []
        self.lengthscales = []
        self.noise = []
        self.mse = []
        self.msll = []
        self.nlpd = []
    def set_train_data(self, train_x, train_y):
        self.train_X = train_x
        self.train_y = train_y
        # print("train y", self.train_y.shape)
        self.model.set_train_data(inputs=train_x, targets=train_y,strict=False)

    def get_fantasy_model(self, new_x, new_y):
        # print("train", self.train_X.shape, "new", new_y.shape)
        self.model = self.model.get_fantasy_model(new_x, new_y)
        

    def fit(self, learning_rate: float = 1e-2, training_iteration: int = 500, verbose: bool = False, debug: bool = False) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)
        self.marginal_log_likelihood = ExactMarginalLogLikelihood(likelihood=self.likelihood,
                                                                                model=self.model)

        for rd in range(training_iteration):
            self.model.train()
            self.likelihood.train()
            self.optimizer.zero_grad()
            output = self.model(self.train_X) ## 
            loss = -self.marginal_log_likelihood(output, self.train_y)
            # print(output.mean, self.train_y, loss)
            loss.backward()
            self.optimizer.step()
            self.model.eval()
            self.likelihood.eval()
            output_from_pred = self.likelihood(self.model(self.train_X))
            mse, msll, nlpd  = evaluation(output_from_pred, self.train_y)
            
            if verbose:
                print('Iter %d/%d - Loss: %.3f  noise: %.3f lengthscale: %.3f mse: %.3f msll: %.3f  nlpd:%.3f' % (
                    rd + 1, training_iteration, loss.item(),
                    self.model.likelihood.noise.item(),
                    self.model.covariance_module.base_kernel.lengthscale.item(),
                    mse,
                    msll,
                    nlpd
                ))
            ##### save the value
            if debug:
                self.lengthscales.append(self.model.covariance_module.base_kernel.lengthscale.item())
                self.noise.append(self.model.likelihood.noise.item())
                self.loss.append(loss.item())
                self.mse.append(mse)
                self.msll.append(msll)
                self.nlpd.append(nlpd)
            ####
        self.lengthscale = self.model.covariance_module.base_kernel.lengthscale.detach()

    def predict(self, test_X: FloatTensor) -> Distribution:
        self.model.eval()
        self.likelihood.eval()
        return self.likelihood(self.model(test_X))

    def compute_posterior_mean_and_covariance_of_data(self, data: FloatTensor):
        predictive_distribution = self.predict(data)
        return predictive_distribution.mean.detach(), predictive_distribution.covariance_matrix.detach()

    def compute_posterior_mean_and_covariance_of_training_data(self):
        return self.compute_posterior_mean_and_covariance_of_data(data=self.train_X)
def evaluation(pred, y_true):
    mse = mean_squared_error(pred, y_true)
    msll = mean_standardized_log_loss(pred, y_true)
    nlpd = negative_log_predictive_density(pred, y_true)
    return mse, msll, nlpd