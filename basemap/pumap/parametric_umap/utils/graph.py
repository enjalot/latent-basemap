import numpy as np
import faiss
from scipy import sparse
import time
import logging
from typing import Tuple, List, Union
import os

import faiss
import numpy as np
from scipy import sparse
import unittest
import torch

def compute_sigma_i(X: np.ndarray, k: int, tol: float = 1e-5, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sigma_i for each sample in the dataset using FAISS for k-nearest neighbors.
    
    This function computes the optimal sigma_i values for each sample that will be used
    in the UMAP probability calculations. It uses binary search to find sigma_i values
    that make the sum of probabilities equal to log2(k).
    
    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_samples, n_features)
    k : int
        Number of nearest neighbors to consider
    tol : float, optional
        Tolerance for binary search convergence, by default 1e-5
    max_iter : int, optional
        Maximum iterations for binary search, by default 100
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - sigma: Computed sigma_i for each sample, shape (n_samples,)
        - rho: Distance to the nearest neighbor for each sample, shape (n_samples,)
        - distances: Euclidean distances to k nearest neighbors, shape (n_samples, k)
        - neighbors: Indices of k nearest neighbors, shape (n_samples, k)
    """
    start_time = time.time()
    logging.info(f"Computing sigma_i for dataset of shape {X.shape} with k={k}")
    
    X = X.astype(np.float32)
    n_samples, n_features = X.shape

    # Enable FAISS multithreading
    faiss.omp_set_num_threads(32)  # Adjust the number based on your CPU cores
    logging.info(f"FAISS multithreading enabled with {faiss.omp_get_max_threads()} threads")
    
    # Step 1: Use FAISS to compute k-nearest neighbors
    logging.info("Building FAISS index...")
    index_start = time.time()
    index = faiss.IndexFlatL2(n_features)
    index.add(X)
    logging.info(f"FAISS index built in {time.time() - index_start:.2f} seconds")

    logging.info("Computing k-nearest neighbors...")
    knn_start = time.time()
    distances_sq, neighbors = index.search(X, k + 1)
    logging.info(f"K-nearest neighbors computed in {time.time() - knn_start:.2f} seconds")

    # Remove self distances and neighbors
    distances_sq = distances_sq[:, 1:].astype(np.float32)  # Shape: (n_samples, k)
    neighbors = neighbors[:, 1:]

    # Convert squared distances to Euclidean distances
    distances = np.sqrt(distances_sq).astype(np.float32)

    # Step 2: Initialize rho and sigma arrays
    rho = distances[:, 0].copy()  # Distance to the nearest neighbor, Shape: (n_samples,)
    target = np.log2(k).astype(np.float32)  # Target sum of probabilities

    # Step 3: Vectorized Binary Search to find sigma_i
    logging.info("Starting binary search for sigma values...")
    binary_search_start = time.time()
    
    # Initialize low and high bounds for all samples
    low = np.full(n_samples, 1e-5, dtype=np.float32)
    high = np.full(n_samples, 10.0, dtype=np.float32)
    sigma = np.zeros(n_samples, dtype=np.float32)

    # Initialize mask to track convergence
    converged = np.zeros(n_samples, dtype=bool)

    for _ in range(max_iter):
        # Compute mid values where not yet converged
        mid = (low + high) / 2.0

        # Compute probabilities: exp(-max(d_ij - rho_i, 0) / sigma_i)
        exponent = -np.maximum(distances - rho[:, np.newaxis], 0) / mid[:, np.newaxis]
        probs = np.exp(exponent)

        # Sum probabilities for each sample
        prob_sum = probs.sum(axis=1)

        # Check convergence
        diff = prob_sum - target
        abs_diff = np.abs(diff)
        newly_converged = (abs_diff < tol) & (~converged)
        converged |= newly_converged

        # Assign sigma where converged
        sigma[newly_converged] = mid[newly_converged]

        # Update high and low based on comparison with the target
        high = np.where(prob_sum > target, mid, high)
        low = np.where(prob_sum <= target, mid, low)

        if _ % 10 == 0:  # Log progress every 10 iterations
            logging.info(f"Binary search iteration {_}: {converged.sum()}/{n_samples} samples converged")

    # For any samples not converged within max_iter, assign the last mid
    sigma[~converged] = mid[~converged]
    logging.info(f"Binary search completed in {time.time() - binary_search_start:.2f} seconds")
    logging.info(f"Total sigma_i computation completed in {time.time() - start_time:.2f} seconds")
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
    sigma, rho, distances, neighbors = compute_sigma_i(X, k, tol, max_iter)

    # Step 2: Compute p_j|i
    P = compute_p_umap(sigma, rho, distances, neighbors)

    # Step 3: Compute symmetric probabilities p^{UMAP}_{ij}
    P_sym = compute_p_umap_symmetric(P)

    logging.info(f"Total UMAP probability computation completed in {time.time() - total_start:.2f} seconds")
    
    if return_dist_and_neigh:
        return P_sym, distances, neighbors
    else:
        return P_sym

logging.info(f"Available CPU cores: {os.cpu_count()}")
logging.info(f"FAISS threads: {faiss.omp_get_max_threads()}")
logging.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
logging.info(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS')}")