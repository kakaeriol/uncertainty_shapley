from joblib import Parallel, delayed
import numpy as np
import torch
from typing import List, Optional, Dict
from collections import defaultdict  # Import defaultdict
import pydvl
from pydvl.parallel import ParallelBackend, _maybe_init_parallel_backend
from pydvl.parallel.config import ParallelConfig
import ot
import gc
from ot import wasserstein_1d
from tqdm.autonotebook import tqdm
from concurrent.futures import wait, FIRST_COMPLETED

from ..dataset import *
# method 1, compute the whole dataset
def label_DS_sw_distance(
    dataset: List[My_Single_Dataset]=[], 
    p: int = 1, 
    projs: torch.Tensor = None, 
    n_projections=100, 
    dict_label: Dict[int, torch.Tensor] = None
):
    """
    Compute label-to-label distances for a dataset using traditional method/sequential method
    Concatinate all of the dataset, and then compute the distance"""
    if projs is None:
        nfts = dataset[0].data.shape[1] 
        projs = torch.tensor(ot.sliced.get_random_projections(nfts, n_projections))
    else:
        n_projections = projs.shape[1]
    if dict_label is None:
        distinct_label = list(set(np.concatenate([i.unique_label for i in dataset])))
        # create label of dataset
        dict_label = {}
        for il in distinct_label:
            dict_label[il] = []
            for ids in dataset:
                dict_label[il].append(ids.data[ids.targets == il] )
            dict_label[il] = torch.concat(dict_label[il])
    else:
        distinct_label = list(dict_label.keys())
    label_distance = np.zeros((len(distinct_label), len(distinct_label)))
    # compute distance between label - only compute ahalf
    for i in range(len(distinct_label)):
         for j in range(i, len(distinct_label)):
             il = distinct_label[i]
             jl = distinct_label[j]
             with torch.no_grad():
                 X_i_proj = dict_label[il]  @ projs
                 X_j_proj = dict_label[jl]  @ projs
                 
                 projected_emd = wasserstein_1d(X_i_proj, X_j_proj, p=p)
                 label_distance[i,j] = (torch.sum(projected_emd) / n_projections) ** (1.0 / p)
                 label_distance[j,i] = label_distance[i,j]
                 del X_i_proj, X_j_proj, projected_emd
                 torch.cuda.empty_cache()  
    del dict_label, projs
    torch.cuda.empty_cache()  # Optionally clear cached memory
    gc.collect() 
    return label_distance


# # method 1 but parallel
# def compute_label_distances_parallel(
#     dataset: List[My_Single_Dataset]=[], 
#     dict_label: Dict[int, torch.Tensor] = None,
#     projs: torch.Tensor = None,
#     n_projections: int = 100,
#     p: int = 1,
#     n_jobs: int = -1  # Use all available cores by default
# ) -> torch.Tensor:
#     """
#     Computes label-to-label distances in parallel.

#     Args:
#         distinct_label: List of all distinct labels.
#         dict_label: Dictionary mapping labels to their corresponding data.
#         projs: Projection matrix.
#         n_projections: Number of projections.
#         p: Power parameter for Wasserstein distance.
#         n_jobs: Number of parallel jobs to use.

#     Returns:
#         torch.Tensor: Matrix of shape (n_labels, n_labels) with computed distances.
#     """
#     if projs is None:
#         nfts = dataset[0].data.shape[1] 
#         projs = torch.tensor(ot.sliced.get_random_projections(nfts, n_projections)).cuda()
#     else:
#         n_projections = projs.shape[1]
#     if dict_label is None:
#         distinct_label = list(set(np.concatenate([i.unique_label for i in dataset])))
#         # create label of dataset
#         dict_label = {}
#         for il in distinct_label:
#             dict_label[il] = []
#             for ids in dataset:
#                 dict_label[il].append(ids.data[ids.targets == il] )
#             dict_label[il] = torch.concat(dict_label[il])
#     else:
#         distinct_label = list(dict_label.keys())
#     def compute_pairwise_distance(
#         i: int,
#         j: int,
#         distinct_label: List[int],
#         dict_label: Dict[int, torch.Tensor],
#         projs: torch.Tensor,
#         n_projections: int,
#         p: float
#     ) -> (int, int, float):
#         """
#         Computes the pairwise distance between two labels.
    
#         Args:
#             i: Index of the first label.
#             j: Index of the second label.
#             distinct_label: List of all distinct labels.
#             dict_label: Dictionary mapping labels to their corresponding data.
#             projs: Projection matrix.
#             n_projections: Number of projections.
#             p: Power parameter for Wasserstein distance.
    
#         Returns:
#             (i, j, distance): Tuple containing indices of the labels and the computed distance.
#         """
#         il = distinct_label[i]
#         jl = distinct_label[j]
#         X_i_proj = dict_label[il] @ projs
#         X_j_proj = dict_label[jl] @ projs
#         projected_emd = wasserstein_1d(X_i_proj, X_j_proj, p=p)
#         distance = (torch.sum(projected_emd) / n_projections) ** (1.0 / p)
#         return i, j, distance
#     n_labels = len(distinct_label)
#     label_distance = torch.zeros((n_labels, n_labels))

#     # Create a list of tasks for parallel execution
#     tasks = [
#         (i, j, distinct_label, dict_label, projs, n_projections, p)
#         for i in range(n_labels)
#         for j in range(i, n_labels)
#     ]
  
#     # Use Joblib to compute distances in parallel
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(compute_pairwise_distance)(*task) for task in tasks
#     )
#     # Fill the distance matrix with the computed distances
#     for i, j, distance in results:
#         label_distance[i, j] = distance
#         label_distance[j, i] = distance  # Symmetric matrix

#     return label_distance

def compute_label_distances_parallel(
    dataset: List[My_Single_Dataset], 
    projs: torch.Tensor = None,
    n_projections: int = 100,
    n_jobs: int = 1,
    p: int = 1,
    parallel_backend: Optional[ParallelBackend] = None,
    progress: bool = False,
):
    """
    Computes label-to-label distances for a dataset in parallel using pyDVL.
    
    Args:
        dataset: dataset include all party, 
        n_jobs: Number of parallel jobs to use.
        parallel_backend: Parallel backend instance for parallelizing computations.
        progress: Whether to display a progress bar.

    Returns:
        np.ndarray: Matrix of shape (n_labels, n_labels) with aggregated distances.
    """
    nfts = dataset[0].data.shape[1] 
    if projs is None:
        projs = torch.tensor(ot.sliced.get_random_projections(nfts, n_projections)).cuda()
    else:
        n_projections = projs.shape[1]
    distinct_label = list(set(np.concatenate([i.unique_label for i in dataset])))
    n_labels = len(distinct_label)
    label_map = {label: index for index, label in enumerate(distinct_label)}
    final_distances = np.zeros((n_labels, n_labels))
    final_counts = np.zeros((n_labels, n_labels))

    # Initialize parallel backend
    parallel_backend = _maybe_init_parallel_backend(parallel_backend, None)
    
    def segment_distance(
        idataset, 
        label_map=label_map, p=1, projs=projs):
        """
        label_map: map label in segment to the indicate label in distinct_label
        """
        n_projections = projs.shape[1]
        n_labels = len(label_map.keys())
        #
        label_distances = np.zeros((n_labels, n_labels))
        label_counts = np.zeros((n_labels, n_labels))
        unique_labels = idataset.unique_label
        # can also parallel this one
        for i in range(len(unique_labels)):
            for j in range(i, len(unique_labels)):
                il = unique_labels[i]
                jl = unique_labels[j]
                X_i_proj = idataset.data[idataset.targets == il] @ projs
                X_j_proj = idataset.data[idataset.targets == jl] @ projs
                projected_emd = wasserstein_1d(X_i_proj, X_j_proj, p=p)
                label_distances[label_map[i], label_map[j]] = (torch.sum(projected_emd) / n_projections) ** (1.0 / p)
                label_distances[label_map[j], label_map[i]] = label_distances[label_map[i], label_map[j]]
                label_counts[label_map[i], label_map[j]] += 1
                label_counts[label_map[j], label_map[i]] += 1
        return label_distances, label_counts

    # Prepare the executor and the progress bar
    max_workers = parallel_backend.effective_n_jobs(n_jobs)
    pbar = tqdm(disable=not progress, total=len(dataset), unit="segments")

    with parallel_backend.executor(max_workers=max_workers, cancel_futures=True) as executor:
        pending: set[Future] = set()
        iterator = iter(dataset)
        
        while True:
            # Refresh progress bar
            pbar.update()

            # Wait for any future to complete
            completed, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                segment_distances, segment_counts = future.result()
                final_distances += segment_distances
                final_counts += segment_counts

            # Ensure that we always have enough jobs running
            try:
                while len(pending) < max_workers:
                    idataset = next(iterator)
                    pending.add(
                        executor.submit(segment_distance, idataset, label_map, p, projs)
                    )
            except StopIteration:
                if len(pending) == 0:
                    break

    # Normalize the aggregated distances by the number of segments or counts
    # This one also can be parallel, let try first
    for i in range(n_labels):
        for j in range(n_labels):
            if final_counts[i, j] > 0:
                final_distances[i, j] /= final_counts[i, j]

    pbar.close()
    return final_distances

# method 2, distributed parallel

def compute_label_distances_distributed(
    dataset: List[My_Single_Dataset], 
    projs: torch.Tensor = None,
    n_projections: int = 100,
    n_jobs: int = 1,
    p: int = 1,
    parallel_backend: Optional[ParallelBackend] = None,
    progress: bool = False,
):
    """
    Computes label-to-label distances for a dataset in parallel using pyDVL.
    
    Args:
        dataset: dataset include all party, 
        n_jobs: Number of parallel jobs to use.
        parallel_backend: Parallel backend instance for parallelizing computations.
        progress: Whether to display a progress bar.

    Returns:
        np.ndarray: Matrix of shape (n_labels, n_labels) with aggregated distances.
    """
    nfts = dataset[0].data.shape[1] 
    if projs is None:
        projs = torch.tensor(ot.sliced.get_random_projections(nfts, n_projections)).cuda()
    else:
        n_projections = projs.shape[1]
    distinct_label = list(set(np.concatenate([i.unique_label for i in dataset])))
    n_labels = len(distinct_label)
    label_map = {label: index for index, label in enumerate(distinct_label)}
    final_distances = np.zeros((n_labels, n_labels))
    final_counts = np.zeros((n_labels, n_labels))

    # Initialize parallel backend
    parallel_backend = _maybe_init_parallel_backend(parallel_backend, None)
    
    def segment_distance(
        idataset, 
        label_map=label_map, p=1, projs=projs):
        """
        label_map: map label in segment to the indicate label in distinct_label
        """
        n_projections = projs.shape[1]
        n_labels = len(label_map.keys())
        #
        label_distances = np.zeros((n_labels, n_labels))
        label_counts = np.zeros((n_labels, n_labels))
        unique_labels = idataset.unique_label
        # can also parallel this one
        for i in range(len(unique_labels)):
            for j in range(i, len(unique_labels)):
                il = unique_labels[i]
                jl = unique_labels[j]
                X_i_proj = idataset.data[idataset.targets == il] @ projs
                X_j_proj = idataset.data[idataset.targets == jl] @ projs
                projected_emd = wasserstein_1d(X_i_proj, X_j_proj, p=p)
                label_distances[label_map[il], label_map[jl]] = (torch.sum(projected_emd) / n_projections) ** (1.0 / p)
                label_distances[label_map[jl], label_map[il]] = label_distances[label_map[il], label_map[jl]]
                label_counts[label_map[il], label_map[jl]] += 1
                label_counts[label_map[jl], label_map[il]] += 1
        return label_distances, label_counts

    # Prepare the executor and the progress bar
    max_workers = parallel_backend.effective_n_jobs(n_jobs)
    pbar = tqdm(disable=not progress, total=len(dataset), unit="segments")

    with parallel_backend.executor(max_workers=max_workers, cancel_futures=True) as executor:
        pending: set[Future] = set()
        iterator = iter(dataset)
        
        while True:
            # Refresh progress bar
            pbar.update()

            # Wait for any future to complete
            completed, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                segment_distances, segment_counts = future.result()
                final_distances += segment_distances
                final_counts += segment_counts

            # Ensure that we always have enough jobs running
            try:
                while len(pending) < max_workers:
                    idataset = next(iterator)
                    pending.add(
                        executor.submit(segment_distance, idataset, label_map, p, projs)
                    )
            except StopIteration:
                if len(pending) == 0:
                    break

    # Normalize the aggregated distances by the number of segments or counts
    # This one also can be parallel, let try first
    for i in range(n_labels):
        for j in range(n_labels):
            if final_counts[i, j] > 0:
                final_distances[i, j] /= final_counts[i, j]

    pbar.close()
    return final_distances





### method 3, sample dataset
def sample_dataset_parallel_dis(
    dataset: List['My_Single_Dataset'], 
    n_jobs: int = -1,
    parallel_backend: Optional['ParallelBackend'] = None,
    progress: bool = False,
    sample_size: int = 500,  # Specify how many samples per segment
    )-> Dict[int, np.ndarray]:

    label_dict = sample_dataset_parallel(dataset, n_jobs, 
                                         parallel_backend,progress, sample_size)
    dis = label_DS_sw_distance(dataset, dict_label=label_dict)
    return dis
    
def sample_dataset_parallel(
    dataset: List['My_Single_Dataset'], 
    n_jobs: int = -1,
    parallel_backend: Optional['ParallelBackend'] = None,
    progress: bool = False,
    sample_size: int = 500,  # Specify how many samples per segment
    )-> Dict[int, np.ndarray]:
    """
    Each label in each dataset, sample the sample_size
    Args:
        dataset: List of dataset segments.
        n_jobs: Number of parallel jobs to use.
        parallel_backend: Parallel backend instance for parallelizing computations.
        progress: Whether to display a progress bar.
        sample_size: Number of samples to draw for each segment.
    
    Returns:
       dictionary array 
    """
    
    # Initialize parallel backend
    parallel_backend = _maybe_init_parallel_backend(parallel_backend, None)
    def map_sample_dataset(
        idataset: List['My_Single_Dataset'],
        sample_size: int = sample_size,
    )->Dict[int, torch.Tensor]:
        """
           sample dataset 
        """
        unique_labels = idataset.unique_label
        # Perform stratified sampling to ensure all labels are represented
        
        sampled_data = {}
        for ilabel in unique_labels:
            sampled_data[ilabel] = []
            label_data = idataset.data[idataset.targets == ilabel]
            if sample_size is None or sample_size > len(label_data):
                # Use all data if sample_size is not defined or greater than available
                sampled_data[ilabel].append(label_data)
            else:
                # Sample with replacement
                sampled_data[ilabel].append(label_data[np.random.choice(len(label_data), sample_size, replace=False)])
        return dict(sampled_data)

    def reduce_sampled_labels(
        sampled_labels_list: List[Dict[int, np.ndarray]]
    ) -> Dict[int, torch.Tensor]:
        """
        Reduces a list of sampled label dictionaries into a single dictionary by appending arrays.
        
        Args:
            sampled_labels_list: List of dictionaries containing sampled data for each label.
        
        Returns:
            Dict[int, np.ndarray]: Combined dictionary of sampled data for each label.
        """
        reduced_data = defaultdict(list)
        
        for sampled_labels in sampled_labels_list:
            for label, data_list in sampled_labels.items():
                reduced_data[label].extend(data_list)  # Append all arrays for this label
        
        # Convert lists of arrays to single concatenated arrays
        for label in reduced_data:
            reduced_data[label] = torch.cat(reduced_data[label], dim=0) 
        
        return dict(reduced_data)

    # Prepare the executor and the progress bar
    max_workers = parallel_backend.effective_n_jobs(n_jobs)
    pbar = tqdm(disable=not progress, total=len(dataset), unit="sampling")
    sampled_labels_list = []
    with parallel_backend.executor(max_workers=max_workers, cancel_futures=True) as executor:
        pending: set[Future] = set()
        iterator = iter(dataset)
        while True:
            pbar.update()
            completed, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                sampled_data = future.result()
                if isinstance(sampled_data, dict):  # Ensure we have the correct type
                    sampled_labels_list.append(sampled_data)
                else:
                    print(f"Unexpected result type: {type(sampled_data)}")  # Debugging output
            # Ensure that we always have enough jobs running
            try:
                while len(pending) < max_workers:
                    idataset = next(iterator)
                    pending.add(
                        executor.submit(map_sample_dataset, idataset, sample_size)
                    )
            except StopIteration:
                if len(pending) == 0:
                    break
    pbar.close()
    final_reduced_data = reduce_sampled_labels(sampled_labels_list)
    
    return final_reduced_data