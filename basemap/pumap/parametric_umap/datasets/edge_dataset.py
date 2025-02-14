import os
import time
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Set, Optional, Iterator, Union
from scipy.sparse import csr_matrix
import pickle

# (Optionally configure logging at the module level)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

class BalancedEdgeBatchIterator:
    """
    Iterator for creating balanced batches of positive and negative edges.
    
    Parameters:
        pos_edges (List[Tuple[int, int]]): The list of positive edges.
        neg_edges (List[Tuple[int, int]]): The list of negative edges.
        pos_ratio (float, optional): Proportion of positive edges per batch (default 0.5).
        batch_size (int, optional): Overall batch size (default 128).
        shuffle (bool, optional): Whether to shuffle the edges at the start of each epoch.
    """
    def __init__(self, pos_edges, neg_edges, pos_ratio=0.5, batch_size=128, shuffle=True):
        # Ensure that the inputs are Python lists (not numpy arrays)
        self.pos_edges = list(pos_edges)
        self.neg_edges = list(neg_edges)
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.shuffle = shuffle
        # Number of positive and negative samples per batch.
        self.num_pos = int(batch_size * pos_ratio)
        self.num_neg = batch_size - self.num_pos

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.pos_edges)
            np.random.shuffle(self.neg_edges)
        # Reset batch indices.
        self.pos_idx = 0
        self.neg_idx = 0
        return self

    def __next__(self):
        # Stop as soon as we've exhausted the positive edges.
        if self.pos_idx >= len(self.pos_edges):
            raise StopIteration

        # Get positive batch.
        pos_batch = self.pos_edges[self.pos_idx:min(self.pos_idx + self.num_pos, len(self.pos_edges))]
        self.pos_idx += self.num_pos

        # If not enough positives remain, sample the remaining with replacement.
        if len(pos_batch) < self.num_pos:
            pos_array = np.empty(len(self.pos_edges), dtype=object)
            pos_array[:] = self.pos_edges
            extra = np.random.choice(pos_array, self.num_pos - len(pos_batch), replace=True)
            if np.isscalar(extra):
                extra = [extra]
            else:
                extra = extra.tolist()
            pos_batch.extend(extra)

        # Get negative batch.
        neg_batch = self.neg_edges[self.neg_idx:min(self.neg_idx + self.num_neg, len(self.neg_edges))]
        self.neg_idx += self.num_neg
        if len(neg_batch) < self.num_neg:
            neg_array = np.empty(len(self.neg_edges), dtype=object)
            neg_array[:] = self.neg_edges
            extra = np.random.choice(neg_array, self.num_neg - len(neg_batch), replace=True)
            if np.isscalar(extra):
                extra = [extra]
            else:
                extra = extra.tolist()
            neg_batch.extend(extra)
        
        batch = pos_batch + neg_batch
        if self.shuffle:
            np.random.shuffle(batch)
        return batch

    def __len__(self):
        # The number of batches is defined solely by the positive edges.
        return int(np.ceil(len(self.pos_edges) / self.num_pos))



# Global variables to hold shared adjacency info in workers.
_global_adj_sets = None
_global_total_nodes = None  # NEW: will store the total number of nodes for candidate list generation

def init_worker(worker_id: int, adj_sets: Dict[int, Set[int]]):
    """
    Initializer for worker processes to set a global variable.
    (Used in single–machine mode, where we build the full adjacency set.)
    """
    logging.info(f"Worker {worker_id} initialization started")
    start_time = time.time()
    global _global_adj_sets
    _global_adj_sets = {node: np.array(list(neighbors)) for node, neighbors in adj_sets.items()}
    logging.info("Worker %d initialization completed in %.2f seconds", worker_id, time.time() - start_time)

def init_worker_partial(worker_id: int, partial_adj_sets: Dict[int, np.ndarray], total_nodes: int):
    """
    NEW: Initializer for a distributed subtask worker that only loads a _partial_
    adjacency dictionary (for the nodes in its chunk) and the total number of nodes.
    """
    logging.info("Worker %d partial initialization started", worker_id)
    start_time = time.time()
    global _global_adj_sets, _global_total_nodes
    _global_adj_sets = partial_adj_sets   # Only for nodes in our chunk.
    _global_total_nodes = total_nodes      # Total nodes in the full graph.
    logging.info("Worker %d partial initialization completed in %.2f seconds; total nodes = %d",
                 worker_id, time.time() - start_time, total_nodes)

def sample_negative_edges_worker(
    node_list: Union[List[int], np.ndarray], 
    k: int, 
    random_state: int
) -> List[Tuple[int, int]]:
    """
    Sample k negative edges for each node in node_list using the global _global_adj_sets.
    Using _global_total_nodes (if set) to create the full candidate list.
    
    Accepts node_list as a NumPy array or a Python list. Each element is cast to an int
    before dictionary lookup to ensure compatibility.
    """
    rng = np.random.RandomState(random_state)
    # Use _global_total_nodes if available; else fall back to the length of _global_adj_sets.
    try:
        total = _global_total_nodes
    except NameError:
        total = len(_global_adj_sets)
    all_nodes = np.arange(total)
    neg_edges = []
    
    # Use tqdm.auto for smart handling of nested progress bars
    for node in tqdm(node_list, desc="Sampling negative edges", leave=False, position=1):
        node_int = int(node)
        # Here, connected is an array of neighbors for node_int (from our partial dictionary).
        connected = _global_adj_sets[node_int]
        mask = ~np.isin(all_nodes, connected)
        mask[node_int] = False
        candidates = all_nodes[mask]
    
        if len(candidates) >= k:
            targets = rng.choice(candidates, size=k, replace=False)
        else:
            targets = candidates
    
        for target in targets:
            neg_edges.append((node_int, int(target)))
    return list(set(neg_edges))

class EdgeBatchIterator:
    """
    Iterator class for batching edges during training.
    
    Parameters
    ----------
    edges : List[Tuple[int, int]]
        List of edges to iterate over
    batch_size : int
        Size of each batch
    shuffle : bool, optional
        Whether to shuffle edges before iteration, by default False
    stratify : bool, optional
        Whether to stratify batches by edge type, by default False
    """
    def __init__(self, edges: List[Tuple[int, int]], batch_size: int, shuffle: bool = False, stratify: bool = False) -> None:
        self.edges = edges
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current: int = 0
        self.current_edges: List[Tuple[int, int]] = self.edges
        self.stratify = stratify
        
    def __iter__(self) -> 'EdgeBatchIterator':
        if self.shuffle:
            # Create a copy to avoid modifying original edges
            self.current_edges = self.edges.copy()
            np.random.shuffle(self.current_edges)
        else:
            self.current_edges = self.edges

        #TODO: add stratify with respect to fake and true edges
        #if self.stratify:

        self.current = 0  # Reset counter when iterator starts
        
        return self
        
    def __next__(self) -> List[Tuple[int, int]]:
        if self.current_edges is None:
            raise StopIteration
            
        if self.current >= len(self.current_edges):
            raise StopIteration
            
        edge_batch = self.current_edges[self.current:self.current + self.batch_size]
        self.current += self.batch_size
        return edge_batch
    
    def __len__(self) -> int:
        return (len(self.current_edges) + self.batch_size - 1) // self.batch_size


class EdgeDataset:
    """
    Dataset class for handling graph edges, including positive and negative edge sampling.
    
    Parameters
    ----------
    P_sym : csr_matrix
        Symmetric probability matrix representing the graph
    """
    def __init__(self, P_sym: csr_matrix) -> None:
        start_time = time.time()
        # For single–machine use we build the full adjacency sets.
        self.adj_sets: Dict[int, Set[int]] = self._get_adjacency_sets_csr(P_sym)
        
        # Obtain positive edges
        P_sym_dok = P_sym.todok()
        self.pos_edges: List[Tuple[int, int]] = list(P_sym_dok.keys())
        
        self.neg_edges: Optional[List[Tuple[int, int]]] = None
        self.all_edges: Optional[List[Tuple[int, int]]] = None
        logging.info("EdgeDataset initialized: %d positive edges, adjacency sets built in %.2f seconds.", 
                     len(self.pos_edges), time.time() - start_time)

    def _shuffle_edges(self, random_state: int = 0) -> None:
        """
        Shuffle all edges using the given random state.
        
        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility, by default 0
        """
        start_time = time.time()
        logging.info("Shuffling all edges...")
        rng = np.random.RandomState(random_state)
        perm = rng.permutation(len(self.all_edges))
        
        # Apply permutation to the edges
        self.all_edges = [self.all_edges[i] for i in perm]
        logging.info("Shuffling complete in %.2f seconds.", time.time() - start_time)
    
    def sample_and_shuffle(self, random_state: int = 0, n_processes: int = 6, verbose: bool = True) -> None:
        """
        Sample negative edges and shuffle all edges.
        
        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility, by default 0
        n_processes : int, optional
            Number of processes for parallel sampling, by default 6
        verbose : bool, optional
            Whether to show progress bars, by default True
        """
        start_time = time.time()
        logging.info("Starting sample_and_shuffle: mixing precomputed negatives (if available) and positives.")
        # Only sample negatives if they have not been provided/precomputed
        if self.neg_edges is None:
            self.sample_negative_edges(random_state=random_state, n_processes=n_processes, verbose=verbose)
        
        self.validate_negative_edges(1000)

        self.all_edges = self.pos_edges + self.neg_edges
        logging.info(f"Total edges: {len(self.pos_edges)} positive + {len(self.neg_edges)} negative")
        self._shuffle_edges(random_state=random_state)
        logging.info("sample_and_shuffle finished in %.2f seconds.", time.time() - start_time)

    def get_loader(self, batch_size: int, sample_first: bool = False, random_state: int = 0, 
                   n_processes: int = 6, verbose: bool = True) -> EdgeBatchIterator:
        """
        Returns an iterator that yields batches of edges.
        
        Parameters
        ----------
        batch_size : int
            Size of each batch
        sample_first : bool, optional
            Whether to sample and shuffle edges before creating loader, by default False
        random_state : int, optional
            Random seed for reproducibility, by default 0
        n_processes : int, optional
            Number of processes for parallel sampling, by default 6
        verbose : bool, optional
            Whether to show progress bars, by default True
            
        Returns
        -------
        EdgeBatchIterator
            Iterator yielding batches of edges
        """
        if sample_first:
            logging.info("get_loader: sample_first=True; sampling and shuffling edges before creating loader.")
            self.sample_and_shuffle(random_state=random_state, n_processes=n_processes, verbose=verbose)
            
        if self.all_edges is None:
            raise ValueError("Must call sample_and_shuffle() before getting loader")
            
        logging.info(f"Creating loader: total {len(self.all_edges)} edges, batch_size={batch_size}.")
        return EdgeBatchIterator(self.all_edges, batch_size)
    
    def sample_negative_edges(self, random_state: int = 0, n_processes: int = 6, verbose: bool = True) -> None:
        """
        Sample negative edges for the graph.
        
        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility, by default 0
        n_processes : int, optional
            Number of processes for parallel sampling, by default 6
        verbose : bool, optional
            Whether to show progress bars, by default True
        """
        start_time = time.time()
        logging.info(f"Sampling negative edges using {n_processes} processes (random_state={random_state})...")
        self.neg_edges = self._sample_negative_edges(
            [src for src, _ in self.pos_edges], 
            random_state=random_state,
            n_processes=n_processes,
            verbose=verbose
        )
        logging.info("Negative edge sampling complete: %d edges sampled in %.2f seconds.",
                     len(self.neg_edges), time.time() - start_time)

    def _sample_negative_edges(self, node_list: List[int], k: int = 5, random_state: int = 0,
                                 n_processes: int = 6, verbose: bool = True) -> List[Tuple[int, int]]:
        """
        Sample k negative edges for each node in parallel.
        
        Parameters
        ----------
        node_list : List[int]
            List of nodes to sample negative edges for
        k : int, optional
            Number of negative edges per node, by default 5
        random_state : int, optional
            Random seed for reproducibility, by default 0
        n_processes : int, optional
            Number of processes for parallel sampling, by default 6
        verbose : bool, optional
            Whether to show progress bars, by default True
            
        Returns
        -------
        List[Tuple[int, int]]
            List of sampled negative edges
        """
        start_time = time.time()
        if n_processes == -1:
            n_processes = os.cpu_count() or 1
        else:
            n_processes = min(n_processes, os.cpu_count() or n_processes)

        base_rng = np.random.RandomState(random_state)
        process_seeds = base_rng.randint(0, np.iinfo(np.int32).max, size=n_processes)

        node_array = np.array(node_list)
        node_chunks = np.array_split(node_array, n_processes)
        logging.info("Dispatching negative edge sampling across %d processes for %d nodes.", n_processes, len(node_list))

        neg_edges = []
        with ProcessPoolExecutor(
            max_workers=n_processes,
            initializer=init_worker,
            initargs=(0, self.adj_sets)
        ) as executor:
            futures = [
                executor.submit(sample_negative_edges_worker,
                                chunk,
                                k,
                                seed)
                for chunk, seed in zip(node_chunks, process_seeds)
            ]
            for future in tqdm(futures, total=len(futures), desc="Completed processes", position=0, leave=True):
                neg_edges.extend(future.result())
        logging.info("Parallel negative edge sampling took %.2f seconds.", time.time() - start_time)
        return neg_edges

    def _get_adjacency_sets(self, P_sym: csr_matrix) -> Dict[int, Set[int]]:
        """
        Get the adjacency set for each node in the graph represented by P_sym.
        
        Parameters
        ----------
        P_sym : csr_matrix
            Symmetric probability matrix
            
        Returns
        -------
        Dict[int, Set[int]]
            Dictionary mapping each node to its set of neighbors
        """
        start_time = time.time()
        n_samples = P_sym.shape[0]
        logging.info("Building adjacency sets for %d nodes...", n_samples)
        adj_sets = []
        
        # Convert to COO format for efficient iteration over non-zero elements
        P_coo = P_sym.tocoo()
        
        # Initialize empty sets for each node
        for _ in range(n_samples):
            adj_sets.append(set())
            
        # Iterate through non-zero elements and add to adjacency sets
        for i, j, val in zip(P_coo.row, P_coo.col, P_coo.data):
            if val > 0:
                adj_sets[i].add(j)
                
        logging.info("Adjacency sets built in %.2f seconds.", time.time() - start_time)
        return {i: set(adj_sets[i]) for i in range(n_samples)}
    
    def _get_adjacency_sets_csr(self, P_sym: csr_matrix) -> Dict[int, Set[int]]:
        start_time = time.time()
        n_samples = P_sym.shape[0]
        logging.info("Building adjacency sets from CSR for %d nodes...", n_samples)
        adj_sets = {}
        for i in range(n_samples):
            start = P_sym.indptr[i]
            end = P_sym.indptr[i + 1]
            neighbors = set(P_sym.indices[start:end])
            adj_sets[i] = neighbors
        logging.info("Adjacency sets built in %.2f seconds.", time.time() - start_time)
        return adj_sets

    def validate_negative_edges(self, sample_size=100):
        import random
        logging.info(f"Validating {sample_size} negative edges...")
        sample = random.sample(self.neg_edges, sample_size)
        num_invalid = 0
        for i, j in sample:
            if j in self.adj_sets.get(i, set()):
                num_invalid += 1
                logging.info(f"Invalid negative: ({i}, {j})")
        logging.info(f"Validation complete. {num_invalid} invalid edges out of {sample_size} sampled.")
    
    def precompute_and_save_negative_edges(self, file_path: str, random_state: int = 0, 
                                             n_processes: int = 6, verbose: bool = True) -> None:
        """
        Precompute negative edges (if not already computed) and save them to disk.
        
        Parameters
        ----------
        file_path : str
            Path where the negative edges will be saved (using pickle).
        random_state : int, optional
            Random seed for reproducibility, by default 0.
        n_processes : int, optional
            Number of processes for parallel sampling, by default 6.
        verbose : bool, optional
            Whether to show progress bars, by default True.
        """
        if self.neg_edges is None:
            logging.info("Negative edges not precomputed. Sampling now...")
            self.sample_negative_edges(random_state=random_state, n_processes=n_processes, verbose=verbose)
        else:
            logging.info("Negative edges already computed. Saving precomputed negatives.")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as f:
            pickle.dump(self.neg_edges, f)
        logging.info("Negative edges saved to %s", file_path)


    @classmethod
    def load_precomputed_negative_edges(file_path: str) -> list:
        """
        Load precomputed negative edges from a file.
        
        Parameters
        ----------
        file_path : str
            Path to the pickle file containing negative edges.
            
        Returns
        -------
        List[Tuple[int, int]]
            Loaded list of negative edges.
        """
        with open(file_path, "rb") as f:
            neg_edges = pickle.load(f)
        return neg_edges

    def get_balanced_loader(self, batch_size: int, pos_ratio: float = 0.5, shuffle: bool = True, random_state: int = 0) -> BalancedEdgeBatchIterator:
        """
        Returns an iterator that yields balanced batches with a fixed positive ratio.
        """
        # Ensure negatives have been sampled; if not, sample and shuffle.
        if self.neg_edges is None:
            self.sample_and_shuffle(random_state=random_state)
        return BalancedEdgeBatchIterator(self.pos_edges, self.neg_edges, pos_ratio=pos_ratio, batch_size=batch_size, shuffle=shuffle)


        

def distributed_sample_negative_edges_subtask(psym_filepath: str, chunk_idx: int, n_chunks: int, 
                                                random_state: int = 0, k: int = 5, verbose: bool = True) -> list:
    # This function is not part of the EdgeDataset class, as it will be run independently in a distributed setting.
    """
    Distributed version: load P_sym from file, determine its own chunk of positive nodes based on the
    provided chunk index and total number of chunks, and sample negative edges for that chunk.
    
    This version is modified to build only a partial adjacency set for the nodes in the current chunk,
    so as to reduce memory usage. The candidate list for negative sampling is still built using the total
    number of nodes.
    
    Parameters
    ----------
    psym_filepath : str
        Path to the precomputed P_sym file.
    chunk_idx : int
        Index of the current chunk (0-indexed).
    n_chunks : int
        Total number of chunks.
    random_state : int, optional
        Random seed for reproducibility, by default 0.
    k : int, optional
        Number of negative edges per node, by default 5.
    verbose : bool, optional
        Whether to run in verbose mode, by default True.
        
    Returns
    -------
    list of Tuple[int, int]
        List of sampled negative edges for the chunk.
    """
    # Load the P_sym matrix from file.
    with open(psym_filepath, "rb") as f:
        loaded = pickle.load(f)
        if isinstance(loaded, dict) and "P_sym" in loaded:
            P_sym = loaded["P_sym"]
        else:
            P_sym = loaded
    total_nodes = P_sym.shape[0]
    if verbose:
        logging.info("Distributed sampling: total nodes = %d", total_nodes)
    
    # Compute the positive source nodes without building the full adjacency set.
    # (Avoiding P_sym.todok() helps reduce memory usage.)
    pos_nodes = [i for i in range(total_nodes) if P_sym.indptr[i] < P_sym.indptr[i+1]]
    if verbose:
        logging.info("Distributed sampling: found %d positive nodes", len(pos_nodes))
    
    # Partition pos_nodes into n_chunks and select the chunk for this subtask.
    pos_nodes_array = np.array(pos_nodes)
    all_chunks = np.array_split(pos_nodes_array, n_chunks)
    if chunk_idx < 0 or chunk_idx >= len(all_chunks):
        raise ValueError(f"Chunk index {chunk_idx} is out of bounds for {n_chunks} chunks.")
    node_chunk = all_chunks[chunk_idx].tolist()
    if verbose:
        logging.info("Chunk %d selected with %d nodes", chunk_idx, len(node_chunk))
    
    # Build a _partial_ adjacency set for only the nodes in the current chunk.
    partial_adj_sets = {}
    for node in node_chunk:
        start = P_sym.indptr[node]
        end = P_sym.indptr[node + 1]
        # The neighbors are stored in the CSR matrix's indices; no need to iterate over all nodes.
        neighbors = P_sym.indices[start:end]
        partial_adj_sets[node] = neighbors
    # Explicitly drop reference to the full P_sym to help garbage collection.
    del P_sym
    
    # Initialize the worker with only the partial adjacency set and the overall node count.
    init_worker_partial(0, partial_adj_sets, total_nodes)
    neg_edges = sample_negative_edges_worker(node_chunk, k, random_state)
    return neg_edges

