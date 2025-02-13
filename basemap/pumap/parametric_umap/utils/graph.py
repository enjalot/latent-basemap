import numpy as np
from scipy import sparse
import time
import logging
from typing import Tuple, List, Union
import os
from tqdm.auto import trange  # use tqdm.auto for nice behavior in notebooks/terminal

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
        for i in range(start, end):
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



# def compute_sigma_i(X: np.ndarray, k: int, tol: float = 1e-5, max_iter: int = 100,
#                     batch_size_nn: int = 100, batch_size_add: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Compute sigma_i for each sample in the dataset using FAISS for k-nearest neighbors.
    
#     This function computes the optimal sigma_i values for each sample that will be used
#     in the UMAP probability calculations. It uses batch-wise nearest neighbor search with FAISS
#     and binary search (with progress reporting) to find sigma_i values that make the sum of probabilities equal to log2(k).
    
#     Parameters
#     ----------
#     X : np.ndarray
#         Dataset of shape (n_samples, n_features)
#     k : int
#         Number of nearest neighbors to consider
#     tol : float, optional
#         Tolerance for binary search convergence, by default 1e-5
#     max_iter : int, optional
#         Maximum iterations for binary search, by default 100
#     batch_size_nn : int, optional
#         Batch size for nearest neighbor search (for progress reporting), by default 10000
#     batch_size_add : int, optional
#         Batch size for adding data to the FAISS index, by default 10000
        
#     Returns
#     -------
#     Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
#         - sigma: Computed sigma_i for each sample, shape (n_samples,)
#         - rho: Distance to the nearest neighbor for each sample, shape (n_samples,)
#         - distances: Euclidean distances to k nearest neighbors, shape (n_samples, k)
#         - neighbors: Indices of k nearest neighbors, shape (n_samples, k)
#     """
#     start_time = time.time()
#     logging.info(f"Computing sigma_i for dataset of shape {X.shape} with k={k}")
    
#     n_samples, n_features = X.shape

#     # Enable FAISS multithreading
#     faiss.omp_set_num_threads(32)
#     logging.info(f"FAISS multithreading enabled with {faiss.omp_get_max_threads()} threads")
#     logging.info(f"Available CPU cores: {os.cpu_count()}")
#     logging.info(f"FAISS threads: {faiss.omp_get_max_threads()}")
#     logging.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
#     logging.info(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS')}")
    
#     # Step 1: Build the FAISS index in batches
#     logging.info("Building FAISS index in batches...")
#     index = faiss.IndexFlatL2(n_features)
#     index_build_start = time.time()
#     for i in trange(0, n_samples, batch_size_add, desc="Adding batches to index"):
#         X_batch = X[i: i + batch_size_add]
#         # Convert each batch to a contiguous array without loading the entire dataset at once
#         X_batch_contig = np.ascontiguousarray(X_batch, dtype=np.float32)
#         index.add(X_batch_contig)
#     logging.info(f"FAISS index built in {time.time() - index_build_start:.2f} seconds")

#     # Step 2: Compute k-nearest neighbors in batches with tqdm progress reporting
#     logging.info("Computing k-nearest neighbors in batches...")
#     knn_start = time.time()
#     all_distances_sq = []
#     all_neighbors = []
#     for i in trange(0, n_samples, batch_size_nn, desc="Nearest neighbor batches"):
#         logging.info(f"Processing batch {i // batch_size_nn} of {n_samples // batch_size_nn}")
#         X_batch = X[i: i + batch_size_nn]
#         X_batch_contig = np.ascontiguousarray(X_batch, dtype=np.float32)
#         logging.info(f"X_batch_contig shape: {X_batch_contig.shape}")
#         distances_sq_batch, neighbors_batch = index.search(X_batch_contig, k + 1)
#         logging.info(f"distances_sq_batch shape: {distances_sq_batch.shape}")
#         logging.info(f"neighbors_batch shape: {neighbors_batch.shape}")
#         all_distances_sq.append(distances_sq_batch)
#         all_neighbors.append(neighbors_batch)
#     distances_sq = np.vstack(all_distances_sq)
#     neighbors = np.vstack(all_neighbors)
#     logging.info(f"K-nearest neighbors computed in {time.time() - knn_start:.2f} seconds")

#     # Remove self distances and neighbors
#     distances_sq = distances_sq[:, 1:].astype(np.float32)  # shape (n_samples, k)
#     neighbors = neighbors[:, 1:]

#     # Convert squared distances to Euclidean distances
#     distances = np.sqrt(distances_sq).astype(np.float32)

#     # Step 3: Compute rho and perform binary search for sigma_i, with tqdm for iterations
#     rho = distances[:, 0].copy()  # nearest neighbor distance for each sample
#     target = np.log2(k).astype(np.float32)  # target sum of probabilities

#     logging.info("Starting binary search for sigma values...")
#     binary_search_start = time.time()
#     low = np.full(n_samples, 1e-5, dtype=np.float32)
#     high = np.full(n_samples, 10.0, dtype=np.float32)
#     sigma = np.zeros(n_samples, dtype=np.float32)
#     converged = np.zeros(n_samples, dtype=bool)

#     for _ in trange(max_iter, desc="Binary search iterations"):
#         mid = (low + high) / 2.0

#         # Compute probabilities
#         exponent = -np.maximum(distances - rho[:, np.newaxis], 0) / mid[:, np.newaxis]
#         probs = np.exp(exponent)
#         prob_sum = probs.sum(axis=1)

#         diff = prob_sum - target
#         abs_diff = np.abs(diff)
#         newly_converged = (abs_diff < tol) & (~converged)
#         converged |= newly_converged
#         sigma[newly_converged] = mid[newly_converged]

#         # Update binary search bounds
#         high = np.where(prob_sum > target, mid, high)
#         low = np.where(prob_sum <= target, mid, low)

#     # For any samples that haven't converged within max_iter, use the last mid value
#     sigma[~converged] = mid[~converged]
#     logging.info(f"Binary search completed in {time.time() - binary_search_start:.2f} seconds")
#     logging.info(f"Total sigma_i computation completed in {time.time() - start_time:.2f} seconds")
    
#     return sigma, rho, distances, neighbors


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


def compute_all_p_umap(X: np.ndarray, k: int, tol: float = 1e-5, max_iter: int = 100,
                      return_dist_and_neigh: bool = False) -> Union[sparse.csr_matrix, Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]]:
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
    return_dist_and_neigh : bool, optional
        Whether to return distances and neighbors, by default False
        
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
    # sigma, rho, distances, neighbors = compute_sigma_i(X, k, tol, max_iter)
    sigma, rho, distances, neighbors = compute_sigma_i_annoy(X, k, tol, max_iter)

    # Step 2: Compute p_j|i
    P = compute_p_umap(sigma, rho, distances, neighbors)

    # Step 3: Compute symmetric probabilities p^{UMAP}_{ij}
    P_sym = compute_p_umap_symmetric(P)

    logging.info(f"Total UMAP probability computation completed in {time.time() - total_start:.2f} seconds")
    
    if return_dist_and_neigh:
        return P_sym, distances, neighbors
    else:
        return P_sym
