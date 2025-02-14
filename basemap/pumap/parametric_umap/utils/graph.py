import numpy as np
from scipy import sparse
import time
import logging
from typing import Tuple, List, Union
import os
from tqdm.auto import trange  # use tqdm.auto for nice behavior in notebooks/terminal
from tqdm import tqdm
from basemap.lancedb_loader import LanceDBLoader

# import faiss
import numpy as np
from scipy import sparse

os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"


def compute_sigma_i_annoy(X: np.ndarray, 
                          k: int, 
                          tol: float = 1e-5, 
                          max_iter: int = 100,
                          n_trees: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sigma_i for each sample in the dataset using the Annoy library for approximate nearest neighbors.
    
    This function computes the optimal sigma_i values for each sample that will be used
    in the UMAP probability calculations, using Annoy to obtain an approximate k-nearest neighbors search.
    It then employs batch-wise processing and binary search to find sigma_i values such that the 
    sum of probabilities is approximately equal to log2(k).
    
    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_samples, n_features)
    k : int
        Number of nearest neighbors (excluding self)
    tol : float, optional
        Tolerance for binary search convergence, by default 1e-5
    max_iter : int, optional
        Maximum iterations for binary search, by default 100
    n_trees : int, optional
        Number of trees to use in Annoy for building the index, by default 10
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - sigma: Computed sigma_i for each sample, shape (n_samples,)
        - rho: Distance to the nearest neighbor for each sample, shape (n_samples,)
        - distances: Euclidean distances to k nearest neighbors, shape (n_samples, k)
        - neighbors: Indices of k nearest neighbors, shape (n_samples, k)
    """
    import time
    import logging
    import os
    from annoy import AnnoyIndex
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import trange, tqdm

    start_time = time.time()
    logging.info(f"Computing sigma_i using Annoy for dataset of shape {X.shape} with k={k}")

    n_samples, n_features = X.shape
    dim = n_features  # Annoy requires the dimension of the vectors

    # Step 1: Build the Annoy index
    logging.info("Building Annoy index...")
    index = AnnoyIndex(dim, metric='dot')
    for i in trange(n_samples, desc="Adding items to Annoy index"):
        # Ensure each item is in float32 format
        index.add_item(i, X[i].astype(np.float32))
    index_build_start = time.time()
    index.build(n_trees)
    logging.info(f"Annoy index built in {time.time() - index_build_start:.2f} seconds")

    # Step 2: Compute k-nearest neighbors using Annoy.
    # Annoy does not have a built-in batch query, so we define a helper
    # function that processes a range of indices. We then use a ThreadPoolExecutor
    # to try to utilize available CPU cores.
    logging.info("Computing k-nearest neighbors using Annoy...")

    num_workers = os.cpu_count() or 1
    # Determine how many indices each worker should process
    chunk_size = (n_samples + num_workers - 1) // num_workers
    logging.info(f"Using {num_workers} workers with chunk size {chunk_size}")

    def get_neighbors_range(start: int, end: int):
        distances_chunk = []
        neighbors_chunk = []
        for i in trange(start, end, desc="Annoy k-NN chunk search"):
            # Get k+1 neighbors. If the index was added normally, the first neighbor
            # should be the query item itself (distance 0).
            nbs, dists = index.get_nns_by_item(i, k + 1, include_distances=True)
            # Remove self if it is present
            if nbs and nbs[0] == i:
                nbs = nbs[1:k+1]
                dists = dists[1:k+1]
            else:
                # In case self is not the first result, remove it if it exists
                if i in nbs:
                    idx_self = nbs.index(i)
                    nbs.pop(idx_self)
                    dists.pop(idx_self)
                nbs = nbs[:k]
                dists = dists[:k]
            distances_chunk.append(dists)
            neighbors_chunk.append(nbs)
        return distances_chunk, neighbors_chunk

    all_distances = []
    all_neighbors = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            futures.append(executor.submit(get_neighbors_range, start, end))
        # Wrap the iteration over futures with tqdm to track progress.
        for future in tqdm(futures, desc="Annoy k-NN search", total=len(futures)):
            d_chunk, n_chunk = future.result()
            all_distances.extend(d_chunk)
            all_neighbors.extend(n_chunk)
            
    # Convert lists to numpy arrays
    distances = np.array(all_distances, dtype=np.float32)  # shape (n_samples, k)
    neighbors = np.array(all_neighbors, dtype=np.int64)      # shape (n_samples, k)
    logging.info("k-nearest neighbors computed using Annoy")

    # Step 3: Compute rho as the distance to the nearest neighbor.
    rho = distances[:, 0].copy().astype(np.float32)

    # Step 4: Perform the binary search for sigma_i (same as in the FAISS version)
    target = np.log2(k).astype(np.float32)
    logging.info("Starting binary search for sigma values...")
    low = np.full(n_samples, 1e-5, dtype=np.float32)
    high = np.full(n_samples, 10.0, dtype=np.float32)
    sigma = np.zeros(n_samples, dtype=np.float32)
    converged = np.zeros(n_samples, dtype=bool)

    for _ in trange(max_iter, desc="Binary search iterations"):
        mid = (low + high) / 2.0
        # Compute probabilities for each neighbor
        exponent = -np.maximum(distances - rho[:, np.newaxis], 0) / mid[:, np.newaxis]
        probs = np.exp(exponent).astype(np.float32)
        prob_sum = probs.sum(axis=1)
        diff = prob_sum - target
        abs_diff = np.abs(diff)
        newly_converged = (abs_diff < tol) & (~converged)
        converged |= newly_converged
        sigma[newly_converged] = mid[newly_converged]
        # Update the bounds based on the condition
        high = np.where(prob_sum > target, mid, high)
        low = np.where(prob_sum <= target, mid, low)

    # Use the last mid value for any samples that did not converge within max_iter.
    sigma[~converged] = mid[~converged]
    total_time = time.time() - start_time
    logging.info(f"Binary search completed in {total_time:.2f} seconds")
    logging.info(f"Total sigma_i computation using Annoy completed in {total_time:.2f} seconds")
    
    return sigma, rho, distances, neighbors



def compute_sigma_i_lance(lancedb_loader, k: int, tol: float = 1e-5, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sigma_i for each sample in the dataset using the lancedb_loader's native search capabilities.

    This function computes the optimal sigma_i values for each sample that will be used
    in the UMAP probability calculations. It leverages Lance's native nearest neighbor search
    and uses multi-threading (via ThreadPoolExecutor) to compute the k-nearest neighbors in parallel.
    A binary search is then performed to determine sigma_i values such that the sum of probabilities
    (computed via an exponential decay) is approximately equal to log2(k).

    Parameters
    ----------
    lancedb_loader : object
        A Lance loader instance that provides:
          - __len__() returning the number of samples,
          - indexable access (lancedb_loader[i]), and
          - a 'search(query, k)' method that returns a tuple:
            (neighbor_indices, distances). Typically, the search returns k+1 results where the first
            entry is the query itself.
    k : int
        Number of nearest neighbors (excluding self).
    tol : float, optional
        Tolerance for binary search convergence; default is 1e-5.
    max_iter : int, optional
        Maximum iterations for the binary search; default is 100.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - sigma: Computed sigma_i for each sample, shape (n_samples,)
        - rho: Distance to the nearest neighbor for each sample, shape (n_samples,)
        - distances: Euclidean distances to k nearest neighbors, shape (n_samples, k)
        - neighbors: Indices of k nearest neighbors, shape (n_samples, k)
    """
    import time
    import logging
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import trange, tqdm

    start_time = time.time()
    n_samples = len(lancedb_loader)
    logging.info(f"Computing sigma_i using Lance for dataset of size {n_samples} with k={k}")

    # Determine number of workers and chunk size
    num_workers = os.cpu_count() or 1
    chunk_size = (n_samples + num_workers - 1) // num_workers
    logging.info(f"Using {num_workers} workers with chunk size {chunk_size}")

    def get_neighbors_range(start: int, end: int):
        distances_chunk = []
        neighbors_chunk = []
        for i in trange(start, end, desc="Lance k-NN chunk search"):
            # Retrieve k+1 neighbors so that we can remove the sample itself
            nn_indices, dists = lancedb_loader.search(lancedb_loader[i], k=k + 1)
            # Remove self if present. Typically the first element is the query itself.
            if nn_indices and nn_indices[0] == i:
                nn_indices = nn_indices[1:k+1]
                dists = dists[1:k+1]
            else:
                # If self is present elsewhere in the list, remove it
                if i in nn_indices:
                    idx_self = nn_indices.index(i)
                    nn_indices.pop(idx_self)
                    dists.pop(idx_self)
                nn_indices = nn_indices[:k]
                dists = dists[:k]
            distances_chunk.append(dists)
            neighbors_chunk.append(nn_indices)
        return distances_chunk, neighbors_chunk

    all_distances = []
    all_neighbors = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            futures.append(executor.submit(get_neighbors_range, start, end))
        for future in tqdm(futures, desc="Lance k-NN search", total=len(futures)):
            d_chunk, n_chunk = future.result()
            all_distances.extend(d_chunk)
            all_neighbors.extend(n_chunk)

    # Convert lists to numpy arrays
    distances = np.array(all_distances, dtype=np.float32)  # shape (n_samples, k)
    neighbors = np.array(all_neighbors, dtype=np.int64)      # shape (n_samples, k)
    logging.info("Nearest neighbors computed using Lance")

    # Compute rho as the distance to the nearest neighbor (first neighbor distance)
    rho = distances[:, 0].copy().astype(np.float32)

    # Binary search to compute sigma_i for each sample such that the sum of probabilities equals log2(k)
    target = np.log2(k).astype(np.float32)
    logging.info("Starting binary search for sigma values (Lance)...")
    low = np.full(n_samples, 1e-5, dtype=np.float32)
    high = np.full(n_samples, 10.0, dtype=np.float32)
    sigma = np.zeros(n_samples, dtype=np.float32)
    converged = np.zeros(n_samples, dtype=bool)

    for _ in trange(max_iter, desc="Binary search iterations (Lance)"):
        mid = (low + high) / 2.0
        # Compute the exponent term: -max(d_ij - rho_i, 0) / mid_i
        exponent = -np.maximum(distances - rho[:, None], 0) / mid[:, None]
        probs = np.exp(exponent).astype(np.float32)
        prob_sum = probs.sum(axis=1)
        abs_diff = np.abs(prob_sum - target)
        newly_converged = (abs_diff < tol) & (~converged)
        converged |= newly_converged
        sigma[newly_converged] = mid[newly_converged]
        # Adjust binary search bounds
        high = np.where(prob_sum > target, mid, high)
        low = np.where(prob_sum <= target, mid, low)

    # For any samples that did not converge, use the last computed mid value.
    sigma[~converged] = mid[~converged]
    total_time = time.time() - start_time
    logging.info(f"Binary search completed in {total_time:.2f} seconds")
    logging.info(f"Total sigma_i computation using Lance completed in {total_time:.2f} seconds")

    return sigma, rho, distances, neighbors



def compute_p_umap(sigma: np.ndarray, rho: np.ndarray, distances: np.ndarray, neighbors: np.ndarray) -> sparse.csr_matrix:
    """
    Compute the conditional probabilities p(UMAP_{j|i}) for each neighbor pair.
    
    This function computes the UMAP conditional probabilities using the formula:
    p(j|i) = exp(-max(d_ij - rho_i, 0) / sigma_i)
    
    Parameters
    ----------
    sigma : np.ndarray
        Computed sigma_i for each sample, shape (n_samples,)
    rho : np.ndarray
        Distance to the nearest neighbor for each sample, shape (n_samples,)
    distances : np.ndarray
        Euclidean distances to k nearest neighbors, shape (n_samples, k)
    neighbors : np.ndarray
        Indices of k nearest neighbors, shape (n_samples, k)
        
    Returns
    -------
    sparse.csr_matrix
        Sparse matrix of conditional probabilities p(j|i)
    """
    start_time = time.time()
    logging.info("Computing UMAP conditional probabilities...")
    
    n_samples, k = distances.shape
    logging.info(f"Processing {n_samples} samples with {k} neighbors each")

    # Ensure sigma has no zero values to avoid division by zero
    sigma = np.maximum(sigma, 1e-10).astype(np.float32)

    # Compute the exponent term: -max(d_ij - rho_i, 0) / sigma_i
    exponent = -np.maximum(distances - rho[:, np.newaxis], 0) / sigma[:, np.newaxis]

    # Compute p_j|i using the exponent
    p_j_i = np.exp(exponent).astype(np.float32)

    # Create a COO sparse matrix
    row_indices = np.repeat(np.arange(n_samples), k)
    col_indices = neighbors.flatten()
    data = p_j_i.flatten()

    P = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_samples))
    P = P.tocsr()  # Convert to CSR format for efficient arithmetic operations

    logging.info(f"Conditional probabilities computed in {time.time() - start_time:.2f} seconds")
    return P

def compute_p_umap_symmetric(P):
    """
    Compute the symmetric UMAP probabilities.
    
    This function computes p^{UMAP}_{ij} using the formula:
    p^{UMAP}_{ij} = p(j|i) + p(i|j) - p(j|i) * p(i|j)
    
    Parameters
    ----------
    P : sparse.csr_matrix
        Matrix of conditional probabilities p(j|i)
        
    Returns
    -------
    sparse.csr_matrix
        Symmetric probability matrix p^{UMAP}_{ij}
    """
    start_time = time.time()
    logging.info("Computing symmetric UMAP probabilities...")
    
    # Compute P + P.T
    P_transpose = P.transpose()
    P_plus_PT = P + P_transpose

    # Compute element-wise multiplication P.multiply(P_transpose)
    P_mul_PT = P.multiply(P_transpose)

    # Compute symmetric probabilities
    P_sym = P_plus_PT - P_mul_PT

    # Optionally, eliminate zeros to maintain sparsity
    P_sym.eliminate_zeros()

    logging.info(f"Symmetric probabilities computed in {time.time() - start_time:.2f} seconds")
    return P_sym


def compute_all_p_umap(X: Union[np.ndarray, LanceDBLoader], k: int, tol: float = 1e-5, max_iter: int = 100) -> sparse.csr_matrix:
    """
    Compute symmetric UMAP probabilities for the entire dataset.
    
    This is a wrapper function that combines the computation of sigma values,
    conditional probabilities, and final symmetric probabilities.
    
    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_samples, n_features)
    k : int
        Number of nearest neighbors
    tol : float, optional
        Tolerance for binary search, by default 1e-5
    max_iter : int, optional
        Maximum iterations for binary search, by default 100
        
    Returns
    -------
    Union[sparse.csr_matrix, Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]]
        If return_dist_and_neigh is False:
            - P_sym: Symmetric probability matrix
        If return_dist_and_neigh is True:
            - P_sym: Symmetric probability matrix
            - distances: Euclidean distances to k nearest neighbors
            - neighbors: Indices of k nearest neighbors
    """
    total_start = time.time()
    logging.info(f"Starting complete UMAP probability computation for dataset of shape {X.shape}")
    
    # Step 1: Compute sigma, rho, distances, and neighbors
    if isinstance(X, LanceDBLoader):
        sigma, rho, distances, neighbors = compute_sigma_i_lance(X, k, tol, max_iter)
    else:
        sigma, rho, distances, neighbors = compute_sigma_i_annoy(X, k, tol, max_iter)

    # Step 2: Compute p_j|i
    P = compute_p_umap(sigma, rho, distances, neighbors)

    # Step 3: Compute symmetric probabilities p^{UMAP}_{ij}
    P_sym = compute_p_umap_symmetric(P)

    logging.info(f"Total UMAP probability computation completed in {time.time() - total_start:.2f} seconds")
    
    return P_sym


def save_umap_results(file_path: str, P_sym, distances=None, neighbors=None):
    """
    Serialize the UMAP results to disk using pickle.

    Parameters
    ----------
    file_path : str
        The file path to save the results.
    P_sym : sparse.csr_matrix
        The symmetric UMAP probability matrix.
    distances : Optional[np.ndarray], optional
        Array of distances to the k nearest neighbors, by default None.
    neighbors : Optional[np.ndarray], optional
        Array of neighbor indices, by default None.
    """
    import pickle
    # Ensure the directory exists                
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    result = {
        "P_sym": P_sym,
        "distances": distances,
        "neighbors": neighbors
    }
    with open(file_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("UMAP results saved to %s", file_path)


def load_umap_results(file_path: str):
    """
    Load serialized UMAP results from disk.

    Parameters
    ----------
    file_path : str
        The file path from which to load the results.
    
    Returns
    -------
    dict
        Dictionary containing the keys "P_sym", "distances", and "neighbors".
    """
    import pickle
    with open(file_path, "rb") as f:
        result = pickle.load(f)
    logging.info("UMAP results loaded from %s", file_path)
    return result


def compute_and_save_all_p_umap(X: Union[np.ndarray, LanceDBLoader], k: int, file_path: str, tol: float = 1e-5,
                                max_iter: int = 100):
    """
    Compute the symmetric UMAP probabilities for the dataset X and serialize the results.
    If the results have already been computed and saved, load them from disk instead.

    Parameters
    ----------
    X : Union[np.ndarray, LanceDBLoader]
        The dataset on which to compute UMAP probabilities.
    k : int
        Number of nearest neighbors.
    file_path : str
        Path to save/load the computed UMAP results.
    tol : float, optional
        Tolerance for binary search convergence, by default 1e-5.
    max_iter : int, optional
        Maximum iterations for binary search, by default 100.

    Returns
    -------
    Union[sparse.csr_matrix, Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]]
        The computed symmetric UMAP probability matrix. If return_dist_and_neigh is True, a tuple
        (P_sym, distances, neighbors) is returned.
    """
    if os.path.exists(file_path):
        logging.info("UMAP results found at %s. Loading from file...", file_path)
        result = load_umap_results(file_path)
    else:
        logging.info("UMAP results not found at %s. Computing UMAP probabilities...", file_path)
        result = compute_all_p_umap(X, k, tol, max_iter)
        save_umap_results(file_path, result)
    return result


