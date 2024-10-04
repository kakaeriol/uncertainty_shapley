from sklearn.manifold import MDS
import ot
from ot import wasserstein_1d
from joblib import Parallel, delayed
import torch
import gc

#
from .. import settings
from .. import utils
from ..kernels import *
# from ..mymodels import *
from ..dataset import *

from typing import List, Optional, Dict, Tuple
from otdd.pytorch.utils import process_device_arg
from tqdm.autonotebook import tqdm
from . import *
import concurrent.futures
#
import pydvl
from pydvl.parallel import ParallelBackend, _maybe_init_parallel_backend
from pydvl.parallel.config import ParallelConfig
from concurrent.futures import wait, FIRST_COMPLETED
def MDS_emb(label_distance, output_dim=2, random_state=0):
    """ 
    Perform MDS embedding on a given distance matrix.
    Input: 
    - label_distance: nxn matrix of label distance
    - output_dim: The number of dimensions for the output embedding. 
    - random_state: A seed for the random. 
    Output: A mapping array nxoutput_dim
    """
    mds = MDS(n_components=output_dim, 
              dissimilarity="precomputed", 
              random_state=random_state)
    embedding = mds.fit_transform(label_distance)
    return embedding
    
def transform_ds(ds, mapping_label, device='cpu'):
    """
    Transform the dataset to a new feature space by concatenating the original data (x) with 
    the mapped label features (mapping_label(y)).
    Input:
        - ds: The input dataset containing 'data' (features) and 'targets' (labels). 
        - mapping_label:  a mapping tensor
        - device (str, optional): The device
    Output: A tensor with the transformed feature space (ds.data|mapping_label(ds.targets))
    """
    x1 = ds.data.to(device,  non_blocking=True)
    if mapping_label is None:
        return x1
    if not isinstance(mapping_label, torch.Tensor):
        mapping_label = torch.tensor(mapping_label).to(device,  non_blocking=True)
    else:
        mapping_label = mapping_label.to(device) 
    y1 = mapping_label[ds.targets].to(device,  non_blocking=True)
    return torch.cat((x1, y1), dim=1)
def SW_distance(x1, x2, projs, p, device="cpu"):
    """
    Compute the Slice Wasserstein Distance between two feature distributions x1 and x2
    Input:
        - x1: input1
        - x2: input2
        - projs: random projection direction
        - p: The order of the Wasserstein distance
        - device: the device
    Output:
        distance
    """
    if projs is None:
        n_projections = 100
        nfts = x1.shape[1]
        projs = torch.tensor(ot.sliced.get_random_projections(nfts, n_projections)).float().to(device)
    else:    
        n_projections = projs.shape[1]
    x1 = x1.to(dtype=torch.double, device=device)
    x2 = x2.to(dtype=torch.double, device=device)
    projs = projs.to(dtype=torch.double, device=device)
    x1_proj = x1 @ projs
    x2_proj = x2 @ projs
    projected_emd = wasserstein_1d(x1_proj,
                                   x2_proj, p=p)
    dis = (torch.sum(projected_emd) / n_projections) ** (1.0 / p)
    return dis
    
def SWEL_distance_ds(d1, d2, mapping_label, projs, p, device="cpu"):
    """
    Compute Slice Wasserstein embedding label distance. 
    Input:
        - d1, d2: two datasets need to compute a distance
        - mapping_label: mapping the label to the embedding space
        - projs: random projection direction
        - p: The order of the Wasserstein distance
        - device: the device
    Output: distance between two dataset
    """
    f1 = transform_ds(d1, mapping_label, device)
    f2 = transform_ds(d2, mapping_label, device)
    dis = SW_distance(f1, f2, projs, p, device)
    return dis

def SWEL_distance(x1_proj, x2_proj, p):
    """ 
    Compute the slice wasersteind distance between 2 embeded proj fts x1, x2
    """
    projected_emd = wasserstein_1d(x1_proj,
                                   x2_proj, p=p)
    dis = (torch.sum(projected_emd) / x1_proj.shape[1]) ** (1.0 / p)
    return dis

def compute_2nint(ic):
    return int(sum([i2*2**i1 for i1,i2 in enumerate(ic)]))



###-------------- swel distance ds index    
def SWEL_distance_ds_index(x1, x2, dataset, mapping_label, projs, p, dict_sym={}, device="cpu"): 
    """
    Compute the SWEL distance between coalitions x1 and x2 in the dataset.

    Inputs:
        - x1, x2: Lists of coalitions.
        - dataset: Dataset from the source party.
        - mapping_label: Mapping tensor array for labels.
        - projs: Random projection directions.
        - p: The order of the Wasserstein distance.
        - dict_sym: Dictionary for memoization to store computed distances.
        - device: The device to run computations on (e.g., "cuda:0").

    Output:
        - rs_dis: A matrix of size n1 x n2 containing the pairwise distances.
    """
    n1 = len(x1)
    n2 = len(x2)
    rs_dis = torch.zeros((n1, n2), dtype=torch.double)
    load_ds = "dsconcat"
    if projs is None:
        n_projections = 100
        nfts = dataset[0].data.shape[1] + mapping_label.shape[1]
        projs = torch.tensor(ot.sliced.get_random_projections(nfts, n_projections)).float().to(device)

    for ii1 in range(n1):
        i1 = x1[ii1]
        start_ii2 = ii1 if x1 is x2 else 0
        for ii2 in range(start_ii2, n2):
            i2 = x2[ii2]
            # print(ii1, ii2)
            # Create a unique key for memoization, taking symmetry into account
            idx_ic = tuple(sorted((compute_2nint(i1), compute_2nint(i2))))
            # Check if the distance has already been computed
            if idx_ic in dict_sym:
                rs_dis[ii1, ii2] = dict_sym[idx_ic]
                continue
            # Convert coalitions to lists
            l_c1 = utils.from_coalition_to_list([i1.numpy()])[0]
            l_c2 = utils.from_coalition_to_list([i2.numpy()])[0]
            # Load datasets for the coalitions
            d1 = load_dataset(l_c1, dataset, load_ds)
            d2 = load_dataset(l_c2, dataset, load_ds)
            # Compute the distance
            distance = SWEL_distance_ds(d1, d2, mapping_label, projs, p, device)
            rs_dis[ii1, ii2] = distance
            if x1 is x2:
                rs_dis[ii2, ii1] = distance
            # Store the computed distance in dict_sym for future reference
            dict_sym[idx_ic] = distance
    return rs_dis


def SWEL_distance_ds_index_mt_chunked(
    x1: List[torch.Tensor],
    x2: List[torch.Tensor],
    dataset: List,
    mapping_label: torch.Tensor,
    projs: torch.Tensor = None,
    p: int = 1,
    n_jobs: int = 1,
    progress: bool = False,
    chunk_size: int = 1000,  # Adjust based on your data size and performance needs
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute the symmetric SWEL distance between coalitions x1 and x2,
    reducing computations by exploiting symmetry and processing in chunks.
    """
    # global dataset
    n1 = len(x1)
    n2 = len(x2)
    
    rs_dis = torch.zeros((n1, n2), dtype=torch.double)
    load_ds = "dsconcat"

    if projs is None:
        n_projections = 100
        nfts = dataset[0].data.shape[1] + mapping_label.shape[1]
        projs = torch.tensor(ot.sliced.get_random_projections(nfts, n_projections)).float().to(device)

    # Prepare arguments for parallel execution in chunks
    pairs = []
    for ii1 in range(n1):
        i1 = x1[ii1]
        start_ii2 = ii1 if x1 is x2 else 0
        for ii2 in range(start_ii2, n2):
            i2 = x2[ii2]
            pairs.append((ii1, ii2, i1, i2))

    # Split pairs into chunks
    chunked_args_list = []
    for start_idx in range(0, len(pairs), chunk_size):
        end_idx = min(start_idx + chunk_size, len(pairs))
        chunk = pairs[start_idx:end_idx]
        chunked_args_list.append((chunk, dataset, mapping_label, projs, p, load_ds, device))

    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
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
    return rs_dis

def compute_distance_pairs_chunk(
    args: Tuple[
        List[Tuple[int, int, torch.Tensor, torch.Tensor]],
        List,
        torch.Tensor,  # mapping_label
        torch.Tensor,  # projs
        int,           # p
        str,           # load_ds
        str            # device
    ]
) -> List[Tuple[int, int, float]]:

    pairs_chunk, dataset, mapping_label, projs, p, load_ds, device = args
    result_chunk = []
    for ii1, ii2, i1, i2 in pairs_chunk:
        with torch.no_grad():
            # Convert coalitions to lists
            l_c1 = utils.from_coalition_to_list([i1])[0]
            l_c2 = utils.from_coalition_to_list([i2])[0]
            # Compute the distance
            # Load datasets for the coalitions using the global dataset
            d1 = load_dataset(l_c1, dataset, load_ds)
            d2 = load_dataset(l_c2, dataset, load_ds)
            distance = SWEL_distance_ds(d1, d2, mapping_label, projs, p, device=device)
            # Store the computed distance
            result_chunk.append((ii1, ii2, distance))

    return result_chunk


def SWEL_distance_ds_index_gpus_chunked(
    x1: List[torch.Tensor],
    x2: List[torch.Tensor],
    dataset,  # The dataset is passed as a parameter
    mapping_label: torch.Tensor,
    projs: torch.Tensor = None,
    p: int = 1,
    n_jobs: int = 1,
    progress: bool = False,
    chunk_size: int = 1000,
    device_ids: List[int] = None,  # List of device IDs for multiple GPUs
) -> torch.Tensor:
    # No need to declare 'global dataset' here

    n1 = len(x1)
    n2 = len(x2)
    rs_dis = torch.zeros((n1, n2), dtype=torch.double)
    load_ds = "dsconcat"

    if projs is None:
        n_projections = 100
        nfts = dataset[0].data.shape[1] + mapping_label.shape[1]
        projs = torch.tensor(ot.sliced.get_random_projections(nfts, n_projections)).float()
        projs = projs.cpu()  # Ensure projs is on CPU before passing to workers

    mapping_label = mapping_label.cpu()  # Ensure mapping_label is on CPU before passing

    # Prepare arguments for parallel execution in chunks
    pairs = []
    for ii1 in range(n1):
        i1 = x1[ii1]
        start_ii2 = ii1 if x1 is x2 else 0
        for ii2 in range(start_ii2, n2):
            i2 = x2[ii2]
            pairs.append((ii1, ii2, i1, i2))

    # Detect available GPUs if device_ids not provided
    if device_ids is None:
        num_gpus = torch.cuda.device_count()
        device_ids = list(range(num_gpus)) if num_gpus > 0 else [0]  # Default to GPU 0

    # Split pairs into chunks and assign device IDs
    chunked_args_list = []
    for idx, start_idx in enumerate(range(0, len(pairs), chunk_size)):
        end_idx = min(start_idx + chunk_size, len(pairs))
        chunk = pairs[start_idx:end_idx]
        # Assign a device ID to this chunk
        device_id = device_ids[idx % len(device_ids)]
        device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        args = (chunk, dataset, mapping_label, projs, p, load_ds, device)
        chunked_args_list.append(args)

    # Use ThreadPoolExecutor for parallel execution
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


def SWEL_distance_ds_index_gpus_chunked_pydvl(
    x1: List[torch.Tensor],
    x2: List[torch.Tensor],
    dataset,  # The dataset is passed as a parameter
    mapping_label: torch.Tensor,
    projs: torch.Tensor = None,
    p: int = 1,
    n_jobs: int = 1,
    chunk_size: int = 1000,
    device_ids: List[int] = None,  # List of device IDs for multiple GPUs
    parallel_backend: Optional[ParallelBackend] = None,
    progress: bool = False,
) -> torch.Tensor:
    
    parallel_backend = _maybe_init_parallel_backend(parallel_backend, None)
    n1 = len(x1)
    n2 = len(x2)
    rs_dis = torch.zeros((n1, n2), dtype=torch.double)
    load_ds = "dsconcat"

    if projs is None:
        n_projections = 100
        nfts = dataset[0].data.shape[1] + mapping_label.shape[1]
        projs = torch.tensor(ot.sliced.get_random_projections(nfts, n_projections)).float()
        projs = projs.cpu()  # Ensure projs is on CPU before passing to workers

    mapping_label = mapping_label.cpu()  # Ensure mapping_label is on CPU before passing

    # Prepare arguments for parallel execution in chunks
    pairs = []
    for ii1 in range(n1):
        i1 = x1[ii1]
        start_ii2 = ii1 if x1 is x2 else 0
        for ii2 in range(start_ii2, n2):
            i2 = x2[ii2]
            pairs.append((ii1, ii2, i1, i2))

    # Detect available GPUs if device_ids not provided
    if device_ids is None:
        num_gpus = torch.cuda.device_count()
        device_ids = list(range(num_gpus)) if num_gpus > 0 else [0]  # Default to GPU 0

    # Split pairs into chunks and assign device IDs
    chunked_args_list = []
    for idx, start_idx in enumerate(range(0, len(pairs), chunk_size)):
        end_idx = min(start_idx + chunk_size, len(pairs))
        chunk = pairs[start_idx:end_idx]
        # Assign a device ID to this chunk
        device_id = device_ids[idx % len(device_ids)]
        device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        args = (chunk, dataset, mapping_label, projs, p, load_ds, device)
        chunked_args_list.append(args)

    max_workers = parallel_backend.effective_n_jobs(n_jobs)
    pbar = tqdm(disable=not progress, total=len(chunked_args_list), unit="chunks")
    rs_all = []
    with parallel_backend.executor(max_workers=max_workers, cancel_futures=True) as executor:
        pending: set[Future] = set()
        iterator = iter(chunked_args_list)
        
        while True:
            pbar.update()
            completed, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                result_chunk = future.result()
                for ii1, ii2, distance in result_chunk:
                    rs_dis[ii1, ii2] = distance
                    if x1 is x2:
                        rs_dis[ii2, ii1] = distance

            # Ensure that we always have enough jobs running
            try:
                while len(pending) < max_workers:
                    idataset = next(iterator)
                    pending.add(
                        executor.submit(compute_distance_pairs_chunk, idataset)
                    )
            except StopIteration:
                if len(pending) == 0:
                    break
            
    return rs_dis

