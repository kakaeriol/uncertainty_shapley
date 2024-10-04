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
from ..kernels import Kernel
from ..utils import utils
from ..mlls import ExactMarginalLogLikelihood
from ..metrics import mean_squared_error, mean_standardized_log_loss

# class ExactGPRegressionModel(gpytorch.models.ExactGP):
class ExactGPRegressionModel(ExactGP):
    def __init__(self, idx_train_x: IntTensor, train_y: FloatTensor, kernel: Kernel,
                 dataset: FloatTensor, 
                 likelihood: GaussianLikelihood, n_player: int, args: argparse.Namespace) -> None:
        super(ExactGPRegressionModel, self).__init__(idx_train_x, train_y, likelihood)
        self.dataset = dataset
        self.mean_module = ConstantMean() # Need to change again!
        self.coalitions = utils.possible_coalitions(n_player)
        self.covariance_module = kernel(dataset=self.dataset, coalitions=self.coalitions, args=args) #my_OTDD_RBF_kernel()
        self.covariance_module.lengthscale = 10 #  10 for moon need to check later
        
        
    def set_new(self, dataset):
        self.train_ds = self.dataset
        self.dataset = dataset
        self.covariance_module.set_newds(dataset)
    def forward(self, x: FloatTensor) -> MultivariateNormal:
        idx = x
        ll = []
        ll_new = []
        for ii1, i1 in enumerate(idx):
            icoalition = self.coalitions[i1].astype(int)
            ll0 = []
            list_all_c = utils.from_coalition_to_list([icoalition])[0]
            # print(icoalition, list_all_c)
            for jj in list_all_c:
                
                si = self.dataset[:, -1] == jj
                X1 = self.dataset[si][:,:-1]
                
               
                # print(X1.shape)
                # print(X1[:, :-1].mean(dim=0).unsqueeze(0).shape)
                # ll.append(X1[:, :-1].mean(dim=0).unsqueeze(0))
                ll0.append(X1)
            # print(torch.cat(ll0, dim=0).shape)
            xnew = torch.cat(ll0, dim=0).mean(dim=0)
            # print(xnew.shape)
            ll.append(xnew.unsqueeze(0))
        new_input_m = torch.cat(ll, dim=0)
        covariance =  self.covariance_module(x)
        # print(new_input_m.shape)
        # print("after", covariance)
        return MultivariateNormal(self.mean_module(new_input_m),
                                 covariance)

    def __call__(self, *args, **kwargs):
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]
        # print("train input: ", train_inputs)
        # print("input:", inputs)
        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            if settings.debug.on():
                if not all(
                    torch.equal(train_input, input) for train_input, input in length_safe_zip(train_inputs, inputs)
                ):
                    raise RuntimeError("You must train on the training inputs!")
            res = super().__call__(*inputs, **kwargs)
            return res

        # Prior mode
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            return full_output

        # Posterior mode
        else:

            if settings.debug.on():
                if all(torch.equal(train_input, input) for train_input, input in length_safe_zip(train_inputs, inputs)):
                    warnings.warn(
                        "The input matches the stored training data. Did you forget to call model.train()?",
                        GPInputWarning,
                    )

            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                # print("strategy prediction")
                train_output = super().__call__(*train_inputs, **kwargs)
                # print("myexGP")
                # print("train input:", train_inputs)
                # print("train output:", train_output)
                # print("train labels:", self.train_targets)
                # print("likelihood:", self.likelihood)
                # Create the prediction strategy for
                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )

            # Concatenate the input to the training input
            # print("batch shape", train_inputs[0].shape[:-2])
            full_inputs = []
            batch_shape = train_inputs[0].shape[:-2]
            for train_input, input in length_safe_zip(train_inputs, inputs):
                # Make sure the batch shapes agree for training/test data
                if batch_shape != train_input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, train_input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                if batch_shape != input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                    input = input.expand(*batch_shape, *input.shape[-2:])
                full_inputs.append(torch.cat([train_input, input], dim=-2))

            # Get the joint distribution for training/test data
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            # full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix
            full_mean, full_covar = full_output.loc, full_output.covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

            # Make the prediction
            with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                (
                    predictive_mean,
                    predictive_covar,
                ) = self.prediction_strategy.exact_prediction(full_mean, full_covar)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)    
        

@dataclass()
class ExactGPRegression_(object):
    train_X: FloatTensor
    train_y: FloatTensor
    kernel: Kernel
    dataset: FloatTensor
    n_player:int
    lengthscale: FloatTensor = field(init=False, default=None)
    optimizer: Optimizer = field(init=False, default=None)
    marginal_log_likelihood: MarginalLogLikelihood = field(init=False, default=None)
    args: argparse.Namespace

    def __post_init__(self):
        self.likelihood = GaussianLikelihood()
        self.model = ExactGPRegressionModel(self.train_X, self.train_y, self.kernel, self.dataset, self.likelihood, self.n_player, self.args)

        # enter train mode
        self.model.train()
        self.likelihood.train()
        self.loss = []
        self.lengthscales = []
        self.noise = []
        self.mse = []
        self.msll = []
        self.model.train()
        self.likelihood.train()

    def fit(self, learning_rate: float = 1e-2, training_iteration: int = 500, verbose: bool = False, debug: bool = False) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)
        self.marginal_log_likelihood = ExactMarginalLogLikelihood(likelihood=self.likelihood,
                                                                                model=self.model)

        for rd in range(training_iteration):
            self.optimizer.zero_grad()
            # print(self.train_X)
            prediction = self.model(self.train_X)
            # return self.marginal_log_likelihood, prediction
            # print(prediction)
            loss = -self.marginal_log_likelihood(prediction, self.train_y)
            loss.backward()
            # pred = prediction.mean.detach().numpy()
            # sigma = prediction.variance.detach().numpy()
            # mse = utils.mse(pred, self.train_y.detach().numpy())
            # msll = utils.msll(pred, self.train_y.detach().numpy(),sigma)
            mse, msll = evaluation(pred, self.train_y)
            if verbose:

                print('Iter %d/%d - Loss: %.3f  noise: %.3f lengthscale: %.3f mse: %.3f msll: %.3f' % (
                    rd + 1, training_iteration, loss.item(),
                    self.model.likelihood.noise.item(),
                    self.model.covariance_module.lengthscale.item(),
                    mse,
                    msll
                ))
            ##### save the value
            if debug:
                self.lengthscales.append(self.model.covariance_module.lengthscale.item())
                self.noise.append(self.model.likelihood.noise.item())
                self.loss.append(loss.item())
                
                self.mse.append(mse)
                self.msll.append(msll)
            ####
            self.optimizer.step()
        self.lengthscale = self.model.covariance_module.lengthscale.detach()

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
    return mse, msll