import numpy as np
from dataclasses import dataclass, field
from torch import FloatTensor
import argparse
from my_gpytorch import utils
from collections import Counter
from my_gpytorch import utils
from my_gpytorch.kernels import Kernel
from my_gpytorch.mymodels import ExactGPScaleRegression_
from copy import deepcopy,copy
import torch
np.random.seed(6)
torch.manual_seed(6)
class SVARM():
    def __init__(self, n_players, budget=None, warm_up=True):
        """budget at least = 2n+2"""
        self.warm_up = warm_up
        if budget is None:
            self.budget = 2*n_players + 2
        else:
            self.budget = budget
        self.n = n_players
        self.Hn = sum([1/s for s in range(1, self.n+1)])
        self.minus_set = []
        self.plus_set  = []
        self.all_players = range(self.n)
        self.check_budget = self.budget
        self.bi_plus_set = None
        self.bi_minus_set = None
    def __conduct_warmup(self):
        for i in self.all_players:
            players_without_i = [j for j in self.all_players if j!=i]
            #sample A_plus
            size_of_A_plus = np.random.choice(self.n, 1)
            A_plus = np.random.choice(players_without_i, size_of_A_plus, replace=False)
            A_plus = np.append(A_plus, i)
            A_plus = sorted(A_plus)
            #sample A_minus
            size_of_A_minus = np.random.choice(self.n, 1)
            A_minus = np.random.choice(players_without_i, size_of_A_minus, replace=False)
            A_minus = np.append(A_minus, i)
            A_minus = sorted(A_minus)
            # if (len(self.minus_set) == 0) or ((len(self.minus_set) > 0) and (A_minus not in self.minus_set)):
            self.minus_set.append(A_minus)
            # if (len(self.plus_set) == 0) or ((len(self.plus_set) > 0) and (A_plus not in self.plus_set)):
            self.plus_set.append(A_plus)
    def __sample_A_plus(self):
        s_plus = np.random.choice(range(1, self.n+1), 1, p=[1/(s*self.Hn) for s in range(1, self.n+1)])
        return np.random.choice(self.all_players, s_plus, replace = False)
    def __sample_A_minus(self):
        s_minus = np.random.choice(range(0, self.n), 1, p=[1/((self.n-s)*self.Hn) for s in range(0, self.n)])
        return np.random.choice(self.all_players, s_minus, replace=False)
    def get_all_budget(self):
        self.__conduct_warmup()
        bi_plus_set = utils.from_list_to_coalitions(self.plus_set, self.n)
        bi_minus_set = utils.from_list_to_coalitions(self.minus_set, self.n)
        # n_current = len(np.unique(np.concatenate([bi_plus_set, bi_minus_set]), axis=0))
        # self.check_budget  = self.budget - n_current
        self.check_budget  = self.budget - 2*self.n
        while self.check_budget > 0:
            A_plus = self.__sample_A_plus()
            A_plus = sorted(A_plus)
            #
            self.plus_set.append(A_plus)
            self.check_budget = self.check_budget  - 1
            # if A_plus not in self.plus_set:
            #     self.plus_set.append(A_plus)
            #     if A_plus not in self.minus_set:
            #         self.check_budget = self.check_budget  - 1
            if self.check_budget  <= 0:
                break
            A_minus = self.__sample_A_minus()
            A_minus = sorted(A_minus)
            self.minus_set.append(A_minus)
            self.check_budget = self.check_budget  - 1
            # if A_minus not in self.minus_set:
            #     self.minus_set.append(A_minus)
            #     if A_minus not in self.plus_set:
            #         self.check_budget = self.check_budget  - 1
        self.bi_plus_set = utils.from_list_to_coalitions(self.plus_set, self.n)
        self.bi_minus_set = utils.from_list_to_coalitions(self.minus_set, self.n)
        return np.unique(np.concatenate([self.bi_plus_set, self.bi_minus_set]), axis=0)

    def compute_shapley(self, d_v_values):
        c_i_plus = np.zeros(self.n)
        phi_i_plus = np.zeros(self.n)
        #
        c_i_minus = np.zeros(self.n)
        phi_i_minus = np.zeros(self.n)

        # warm up phrase:
        for i in range(self.n):
            ## plus
            value = d_v_values[frozenset(self.plus_set[i])]
            phi_i_plus[i] = value
            c_i_plus[i] = 1

            ## minus
            value = d_v_values[frozenset(self.minus_set[i])]
            phi_i_minus[i] = value
            c_i_minus[i] = 1
            
        
        ## update positive: 
        for ii in range(self.n, len(self.plus_set)):
            A = self.plus_set[ii]
            value = d_v_values[frozenset(A)]
            for i in A:
                phi_i_plus[i] = (phi_i_plus[i]*c_i_plus[i] + value)/(c_i_plus[i] + 1)
                c_i_plus[i] = c_i_plus[i] + 1

        for ii in range(self.n, len(self.minus_set)):
            A = self.minus_set[ii]
            value = d_v_values[frozenset(A)]
            for i in A:
                phi_i_minus[i] = (phi_i_minus[i]*c_i_minus[i] + value)/(c_i_minus[i] + 1)
                c_i_minus[i] = c_i_minus[i] + 1

        shapley = phi_i_plus - phi_i_minus
        # # normalize:
        # grand_coalition = 0.999
        # shapley *= (grand_coalition / np.sum(shapley))
        return shapley
                
        

@dataclass
class uncertainty_framework(object):
    dataset: FloatTensor
    T: FloatTensor
    kernel: Kernel
    n_random: int
    n_active: int
    args: argparse.Namespace
    v_T: FloatTensor = None
    testset: FloatTensor = None
    
    def __post_init__(self):
        self.n_predict = len(self.T) - self.n_random - self.n_active
        idx = range(len(self.T))
        if self.n_random > 0:
            self.indx_rd = np.random.choice(idx, self.n_random, replace=False)   
            self.add_train_item = list(self.indx_rd)
            X_train = self.T[self.indx_rd]
            y_train = self._get_vT(self.indx_rd)
            self.my_model = ExactGPScaleRegression_(X_train, y_train, self.kernel, self.dataset,args=self.args)
            self.my_model.fit(learning_rate=self.args.learning_rate, training_iteration=self.args.training_interation,verbose=True, debug=True)
        

    def _get_vT(self, idx):
        """ If already compute v_T just return it,
            else will write later
            else:
            build XGBOOST model with idx and then compute model performance on test
            """
        if self.v_T is None:
            # will add later
            return None
        else:
            return self.v_T[idx]

    def compute_uncertainty(self, method="fantasy"):
        """There is three way to train: 'fantasy' / fast, build in from gpytorch, 
            'update_train': update data, and  'scratch': retrain from scratch,
            default is fantasy
            """
        T = self.T
        idx = range(len(T))
        if self.n_random > 0:
            remaining = [i for i in idx if i not in self.indx_rd]
            update_model = copy(self.my_model)
            add_train_item = self.add_train_item
        else:
            remaining = idx
        ### --- 
        for i in range(self.n_active):
            if (self.n_random == 0) & (i==0): 
                # begin with the one that has a lot of parties
                index_ = torch.argmax(T.sum(axis=1)).item()
                X_train = T[index_].expand(1, -1)
                y_train = self._get_vT(index)
                self.my_model = ExactGPScaleRegression_(X_train, y_train, self.kernel, self.dataset,args=self.args)
                self.my_model.fit(learning_rate=self.args.learning_rate, training_iteration=self.args.training_interation,verbose=True, debug=True)
                update_model = copy(self.my_model)
                self.add_train_item = list(index)
                add_train_item = self.add_train_item
                
                continue
            obs = update_model.predict(T[remaining])
            variance = obs.variance
            #
            max_variance = variance.detach().numpy().max()
            index_variance = np.where(variance.detach().numpy() == max_variance)[0][0]
            # add_train_item = item_test_list[index_variance]
            new_item = remaining[index_variance]
            remaining = [ii for ii in remaining if ii != new_item]
            add_train_item.append(new_item)
            # 
            if method == 'fantasy':
                X_train_add = T[new_item].expand(1,-1)
                y_train_add = self._get_vT(new_item).unsqueeze(0)
                update_model.get_fantasy_model(X_train_add, y_train_add)
            elif method == 'update_train':
                X_train_add = self.T[add_train_item]
                y_train_add = self._get_vT(add_train_item)
                update_model.set_train_data(X_train_add, y_train_add)
                update_model.fit(learning_rate=args.learning_rate, training_iteration=10,verbose=False, debug=False)
            elif method == 'scratch':
                X_train_add = T[add_train_item]
                y_train_add = self._get_vT(add_train_item)
                update_model = ExactGPScaleRegression_(X_train_add, y_train_add, kernel, dataset,args=args)
                update_model.fit(learning_rate=args.learning_rate, training_iteration=args.training_interation,verbose=False, debug=False)

        X_predict = T[remaining]
        y_predict = T[remaining]
        v_pred = update_model.predict(X_predict)
        all_T = np.hstack([self._get_vT(add_train_item),v_pred.mean.detach().numpy()])
        all_variance = np.hstack([np.zeros_like(self._get_vT(add_train_item)),v_pred.variance.detach().numpy()])
        items = add_train_item + remaining
        return items, all_T, all_variance, update_model, y_predict

