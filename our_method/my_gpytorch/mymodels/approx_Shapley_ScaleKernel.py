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
from ..models import ApproximateGP
from ..variational import CholeskyVariationalDistribution, VariationalStrategy
from ..mlls.marginal_log_likelihood import MarginalLogLikelihood
from ..means import ConstantMean
from ..distributions import Distribution
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
from xgboost import XGBClassifier
#
from torch import FloatTensor
from torch import LongTensor
from torch import IntTensor
from torch import BoolTensor
from ..means import ConstantMean
from ..models import ExactGP
from ..kernels import Kernel
from ..kernels import ScaleKernel
from ..utils import utils
from ..mlls import ExactMarginalLogLikelihood
#
from ..constraints import constraints
from ..priors import NormalPrior,GammaPrior
from ..metrics import mean_squared_error, mean_standardized_log_loss,negative_log_predictive_density
from . import ExactGPScaleRegression_

class approx_Shapley_ScaleKernel(ExactGPScaleRegression_):
    def __init__(self,
        kernel: Kernel,
        dataset: FloatTensor,
        D_test: FloatTensor,
        n_player:int,
        args: argparse.Namespace):
            super(approx_Shapley_ScaleKernel, self).__init__(None, None, kernel, dataset, n_player, args)
            self.D_test = D_test

    def __post_init__(self):
        super(approx_Shapley_ScaleKernel, self).__post_init__()
        self.coalitions = utils.possible_coalitions(self.n_player)
        
    def uncertainty_shapley_framework(self, T:int, method: str = "random", J:int=100, learning_rate: float = 1e-2, training_iteration: int = 500, verbose: bool = False, debug: bool = False, v_values=None) -> None:
        """ """
        indx_rd = []
        if method == "random":
            indx_rd = np.random.choice(range(len(coalitions)), size=T, replace=False)
            id_training = coalitions[indx_rd]
            idx = torch.tensor(indx_rd, dtype=torch.int)
            idx = idx.reshape(-1,1)
            # ------------------------------------------------------------ 
            # COMPUTE Y TRAIN #
            if v_values is not None: # already compute and save as v_values
                y_train = torch.tensor(v_values[indx_rd])
            else:
                v_values = []
                for ic in id_training:
                    v_values.append(get_v_values(ic, self.dataset, self.D_test))
                y_train = torch.tensor(v_values, dtype=torch.float64)
            self.set_train_data(idx, y_train)
            print("trainX:",self.train_X, self.train_y)
            self.fit(learning_rate, training_iteration, verbose, debug)
            remaining = [i for i in range(len(self.coalitions)) if i not in self.train_X]
        else:
            item_test_list = range(len(self.coalitions))
            for t  in range(T):
                if method == "uncertainty":
                    if t == 0:
                        indx_rd = np.random.choice(item_test_list, 1, replace=False)   
                    else:
                        remaining = [i for i in item_test_list if i not in indx_rd]
                        variance = observed_pred.variance
                        max_variance = variance.detach().numpy()[remaining].max()
                        index_variance = np.where(variance[remaining].detach().numpy() == max_variance)[0][0]
                        add_train_item = remaining[index_variance]
                        indx_rd = np.append(indx_rd, add_train_item)
                else: #mutal information method
                    if t == 0:
                        idx_pred = torch.tensor(item_test_list, dtype=torch.int)
                        prior = self.predict(idx_pred)
                        variance =  prior.variance
                        max_variance = variance.detach().numpy().max()
                        index_variance = np.where(variance.detach().numpy() == max_variance)[0][0]
                        add_train_item = item_test_list[index_variance]
                        indx_rd.append(add_train_item)
                    else:
                        remaining = [i for i in item_test_list if i not in indx_rd]
                        All = observed_pred.covariance_matrix
                        observed_variance = observed_pred.variance
                        A_train = All[indx_rd][:,indx_rd] 
                        A_obs   = All[remaining][:, remaining]
                        # ----- 
                        variance_train = observed_variance[indx_rd]
                        variance_obs = observed_variance[remaining]
                        #
                        num_ = observed_variance - torch.diag(torch.matmul(torch.matmul(All[:][:,indx_rd], torch.inverse(A_train)), All[indx_rd,:][:]))
                        dem_ = observed_variance - torch.diag(torch.matmul(torch.matmul(All[:][:,remaining], torch.inverse(A_obs)), All[remaining,:][:]))
                        del_ta = num_/dem_

                        # num_ = observed_variance - torch.matmul(torch.matmul(All[:][:,indx_rd], torch.inverse(A_train)), All[indx_rd,:][:])
                        # dem_ = observed_variance - torch.matmul(torch.matmul(All[:][:,remaining], torch.inverse(A_obs)), All[remaining,:][:])
                        # del_ta = torch.div(num_, dem_)
                        # print("delta", del_ta.shape, del_ta)
        
                        # max_delta = del_ta.detach().numpy()[remaining]).max()
                        delta_temp = del_ta.detach().numpy()[remaining]
                        max_delta = delta_temp[np.isfinite(delta_temp)].max()
                        # index_delta = np.where(del_ta[remaining].detach().numpy() == max_delta)[0][0]
                        # add_train_item = remaining[index_delta]
                        index_delta = np.where(del_ta.detach().numpy() == max_delta)[0][0]
                        add_train_item = item_test_list[index_delta]
                        indx_rd.append(add_train_item)
                        # print("indx_rd", indx_rd)
              ### set xtrain, ytrain
                id_training = coalitions[indx_rd]
                idx = torch.tensor(indx_rd, dtype=torch.int)
                idx = idx.reshape(-1,1)
                # ------------------------------------------------------------ 
                # COMPUTE Y TRAIN #
                if v_values is not None: # already compute and save as v_values, need to think and fix later
                    y_train = torch.tensor(v_values[indx_rd])
                else: 
                    v_values = []
                    for ic in id_training:
                        v_values.append(get_v_values(ic, self.dataset, self.D_test))
                    y_train = torch.tensor(v_values, dtype=torch.float64)
                
                self.set_train_data(idx, y_train)       
                self.fit(learning_rate, training_iteration, verbose, debug) ### think to add loop for greedy here!
                idx_pred = torch.tensor(item_test_list, dtype=torch.int)
                observed_pred = self.predict(idx_pred)   
                        
                            
        ## --- compute uncertainty shapley
        if len(remaining) > 0: 
            test_idx = torch.tensor(remaining, dtype=torch.int)
            test_idx = test_idx.reshape(-1,1)
            val = self.predict(test_idx)
            #---
            A  = pd.DataFrame(columns=range(len(self.coalitions)))
            A[remaining] = val.sample(torch.Size((J,)))
            A[indx_rd] = y_train.numpy()        
            self.shapley_sampling = A.apply(lambda x: pd.Series(Shapley_value(x, n_player)), axis=1)
            self.predict = val
            self.shapley_ess = self.shapley_sampling.mean(axis=0).to_numpy()
            self.shapley_variance = self.shapley_sampling.std(axis=0).to_numpy()**2
        else:      
            sort = np.argsort(indx_rd)   
            self.shapley_sampling = None
            self.shapley_ess = Shapley_value(y_train[sort], n_player)
            self.shapley_variance = np.zeros_like(self.shapley_ess)
            

        
        