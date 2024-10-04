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
from ..label_distance import *

import gpytorch
import ot
from ot import wasserstein_1d


import gpytorch
import ot
from ot import wasserstein_1d


from threading import Lock
def compute_distance_pairs_chunk(
    args:Tuple[
    Tuple[int, int, torch.Tensor, torch.Tensor], # index, index, i, j
     List, # dataset
     int, # p
     str, # device
    dict, # sym_dict
    Lock, # lock
    
    ])-> List[Tuple[int, int, float]]:
    pairs_chunk, dataset, p, device, sym_dict, lock = args
    result_chunk = []
    for ii1, ii2, ic1, ic2 in pairs_chunk:
        key = frozenset((compute_2nint(ic1), compute_2nint(ic2)))
        with lock:
            if key in sym_dict.keys():
                dis = sym_dict[key]
                result_chunk.append((ii1, ii2, dis))
                continue
        with torch.no_grad():
            x_proj = load_data(ic1, dataset).to(device)
            y_proj = load_data(ic2, dataset).to(device)
            dis = SWEL_distance(x_proj, y_proj, p).item()
            result_chunk.append((ii1, ii2, dis))
        with lock:
            sym_dict[key] = dis
    return result_chunk
            
def SWEL_distance_ds_index_gpus_chunked(
    x1, x2, dataset, p, chunk_size = 50, 
    device_ids=None, n_jobs: int = 1, 
    sym_dict = {}, lock = None, 
    progress: bool = False,
   
):
    """ """
    n1 = len(x1)
    n2 = len(x2)
    rs_dis = torch.zeros((n1, n2), dtype=torch.double)
    pairs = []
    for ii1 in range(n1):
        i1 = x1[ii1]
        start_ii2 = ii1 if x1 is x2 else 0
        for ii2 in range(start_ii2, n2):
            i2 = x2[ii2]
            pairs.append((ii1, ii2, i1, i2))
    if device_ids is None:
        num_gpus = torch.cuda.device_count()
        device_ids = list(range(num_gpus)) if num_gpus > 0 else [0]
    #    
    chunked_args_list = []
    for idx, start_idx in enumerate(range(0, len(pairs), chunk_size)):
        end_idx = min(start_idx + chunk_size, len(pairs))
        chunk = pairs[start_idx:end_idx]
        # Assign a device ID to this chunk
        device_id = device_ids[idx % len(device_ids)]
        device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        args = (chunk, dataset, p, device, sym_dict, lock)
        chunked_args_list.append(args)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # Submit tasks individually
        futures = {
            executor.submit(compute_distance_pairs_chunk, args): args
            for args in chunked_args_list
        }
        # Process tasks as they complete
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            disable=not progress,
            desc="Computing distances",
        ):
            result_chunk = future.result()
            for ii1, ii2, distance in result_chunk:
                rs_dis[ii1, ii2] = distance
                if x1 is x2:
                    rs_dis[ii2, ii1] = distance

            del result_chunk
            gc.collect()
    del future
    # Ensure all threads are done
    executor.shutdown(wait=True)
    gc.collect()
    torch.cuda.empty_cache()
    return rs_dis

# Parallel class 
def compute_2nint(ic):
    # Ensure ic is a list or numpy array
    ic_list = ic.cpu().numpy().tolist() if isinstance(ic, torch.Tensor) else ic
    return int(sum([i2 * 2 ** i1 for i1, i2 in enumerate(ic_list)]))


def SWEL_distance(x1_proj, x2_proj, p):
    """ 
    Compute the slice wasersteind distance between 2 embeded proj fts x1, x2
    """
    projected_emd = wasserstein_1d(x1_proj,
                                   x2_proj, p=p)
    dis = (torch.sum(projected_emd) / x1_proj.shape[1]) ** (1.0 / p)
    return dis
    
def SWEL_distance_index(i1, i2, dataset, p):
    """ """
    x1_proj = load_data(i1, dataset)
    x2_proj = load_data(i2, dataset)
    return SWEL_distance(x1_proj, x2_proj, p)

def load_data(i1, dataset):
    idx = utils.from_coalition_to_list([i1.cpu().numpy()])[0]
    lds = [dataset[i] for i in idx]
    return torch.concat(lds)
    
class Exponential_SW_Kernel(Kernel):
    has_lengthscale = True
    def __init__(self, dataset, args, **kwargs):
        """
        Custom kernel for Gaussian Processes using the exponential of the SWEL distance.
        
        :param dataset: The dataset to compute the SWEL distance on.
        :param mapping_label: The label mapping tensor for computing SWEL.
        :param lengthscale_prior: Optional prior for the lengthscale parameter.
        :param kwargs: Additional arguments for the GPyTorch Kernel.
        """
        super(Exponential_SW_Kernel, self).__init__(has_lengthscale=True, active_dimes=None, **kwargs)
        # Store the dataset and mapping label
        self.dataset = dataset
        self.args = args
        self.p = vars(self.args).get("p", 2)
        self.sym_dict = {}
        
    def forward(self, x1, x2, diag=False, **params):
        """
        Forward function for the custom kernel.
        This computes the kernel matrix between x1 and x2 using the exponential of the SWEL distance.
        """
        lengthscale = self.lengthscale
        p = self.args.p
        if self.args.device_ids is not None:
            chunk_size = self.args.chunk_size
            device_ids = self.args.device_ids
            n_jobs = self.args.n_jobs
            import threading
            lock = threading.Lock()  # Create a lock for thread safet
            # Ensure lengthscale is positive
            
            #
            distances = SWEL_distance_ds_index_gpus_chunked(x1, x2, self.dataset, self.args.p, 
                                                            chunk_size, device_ids, n_jobs, 
                                                            sym_dict = self.sym_dict, lock = lock, 
                                                           )
        else:
            device = self.args.device
            n1 = x1.shape[0]
            n2 = x2.shape[0]
            distances = torch.zeros((n1, n2),  dtype=torch.double)
            for ii1, ic1 in enumerate(x1):
                for ii2, ic2 in enumerate(x2):
                    key = frozenset((compute_2nint(ic1), compute_2nint(ic2)))
                    if key in self.sym_dict.keys():
                        distances[ii1, ii2] = self.sym_dict[key]
                        continue

                    if torch.equal(ic1, ic2):
                        distances[ii1, ii2] = 0
                        continue
                    x_proj = load_data(ic1, self.dataset).to(device)
                    y_proj = load_data(ic2, self.dataset).to(device)
                    distances[ii1, ii2] = SWEL_distance(x_proj, y_proj, p).item()
                    self.sym_dict[key] = distances[ii1, ii2]
                        
            
        # Compute the exponential kernel using the SWEL distance
        # exp_kernel = torch.exp(-distances / (2 * lengthscale ** 2))
        exp_kernel = torch.exp(-distances / (lengthscale ** 2))
        # exp_kernel = torch.exp(-distances / (lengthscale ))

    
        # Return the kernel matrix
        return exp_kernel    

