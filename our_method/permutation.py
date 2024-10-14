import ot
import otdd
import numpy as np
from otdd.pytorch.distance import DatasetDistance
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import argparse
import sys
#
import math
sys.path.append("/external1/np/euler/Develop_uncertainty/our_method")
from my_gpytorch import settings
from my_gpytorch import utils
from my_gpytorch.kernels import *
from my_gpytorch.mymodels import *
from uncertainty_framework import uncertainty_framework
from my_gpytorch.dataset import *
from my_gpytorch.mymodels import *
from net import *
import pdb
import dill
from joblib import Parallel, delayed
import multiprocessing 
class PermutationSampling:
    def __init__(self, n_players, budget=512):
        """
        Initialize the PermutationSampling class.

        Parameters:
        - n_players: int, number of players in the game.
        - budget: int, maximum number of unique coalitions to sample.
        """
        self.permutations = []
        self.coalitions = []        # List to maintain the order of coalitions
        self.coalitions_set = set() # Set for O(1) lookup to ensure uniqueness
        self.n_players = n_players
        self.budget = budget
        self.shapley_values = np.zeros(n_players)
        if budget > 0:
            self.__init_all_samples_for_Shapley()
        
    def set_permutation(self, permutation
                        ,coalition):
        self.permutations = permutation
        self.coalitions  = coalition


    def set_frozen_set_v_values(self, frozen_set_dict):
        self.frozen_set_dict = frozen_set_dict
        
    def __init_all_samples_for_Shapley(self):
        """
        Generate unique coalitions by sampling permutations.
        """
        players = list(range(self.n_players))
        print(f"Players: {players}")
        while len(self.coalitions) < self.budget:
            permutation = list(np.random.permutation(players))
            self.permutations.append(permutation)
            coalition = set()
            for player in permutation:
                coalition.add(player)
                # Convert coalition to frozenset for hashing
                ic = frozenset(coalition)
                if ic not in self.coalitions_set:
                    self.coalitions.append(list(ic))
                    self.coalitions_set.add(ic)
                    # Check if we've reached the budget
    def comput_shapley_values(self):
        shapley_values = self.shapley_values
        for permutation in self.permutations:
            coalition = set()
            prev_value = 0
            for player in permutation:
                coalition.add(player)
                # print(coalition)
                ic_set = frozenset(coalition)
                curr_value = self.frozen_set_dict[ic_set]
                # Marginal contribution of the player
                marginal_contribution = curr_value - prev_value
                shapley_values[player] += marginal_contribution
                prev_value = curr_value
        self.shapley_values=shapley_values/len(self.permutations)

    def Shapely_adding_permutation(self, more_budget):
        coalitions = self.coalitions
        nperm = len(self.permutations)
        shapley = self.shapley_values*nperm
        count=0
        while True:
            permutation = list(np.random.permutation(self.n_players))
            coalition = set()
            prev_value = 0
            count+=1
            for player in permutation:
                coalition.add(player)
                ic_set = frozenset(coalition)
                curr_value = self.frozen_set_dict[ic_set]
                if ic_set not in coalitions:
                    more_budget -= 1
                marginal_contribution = curr_value - prev_value
                shapley[player] += marginal_contribution
                prev_value = curr_value
            if more_budget <= 0:
                return shapley/(count+nperm)
