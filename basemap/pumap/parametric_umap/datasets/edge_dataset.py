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

# Global variable to hold shared adjacency sets in workers
_global_adj_sets = None

def init_worker(worker_id: int, adj_sets: Dict[int, Set[int]]):
    """
    Initializer for worker processes to set a global variable.
    """
    logging.info(f"Worker {worker_id} initialization started")
    start_time = time.time()
    global _global_adj_sets
    _global_adj_sets = {node: np.array(list(neighbors)) for node, neighbors in adj_sets.items()}
    logging.info(f"Worker {worker_id} initialization completed in %.2f seconds", time.time() - start_time)

def sample_negative_edges_worker(
    node_list: Union[List[int], np.ndarray], 
    k: int, 
    random_state: int
) -> List[Tuple[int, int]]:
    """
    Sample k negative edges for each node in node_list using the global _global_adj_sets.
    
    Accepts node_list as a NumPy array or a Python list. Each element is cast to an int
    before dictionary lookup to ensure compatibility.
    """
    rng = np.random.RandomState(random_state)
    n_nodes = len(_global_adj_sets)
    all_nodes = np.arange(n_nodes)
    neg_edges = []
    
    # Use tqdm.auto for smart handling of nested progress bars
    for node in tqdm(node_list, desc="Sampling negative edges", leave=False, position=1):
        node_int = int(node)
        # Here, connected is already a NumPy array
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
        # self.adj_sets: Dict[int, Set[int]] = self._get_adjacency_sets(P_sym)
        self.adj_sets: Dict[int, Set[int]] = self._get_adjacency_sets_csr(P_sym)
        
        # Obtain positive edges
        P_sym_dok = P_sym.todok()
        self.pos_edges: List[Tuple[int, int]] = list(P_sym_dok.keys())
        
        self.neg_edges: Optional[List[Tuple[int, int]]] = None
        self.all_edges: Optional[List[Tuple[int, int]]] = None
        logging.info(f"EdgeDataset initialized: {len(self.pos_edges)} positive edges, adjacency sets built in {time.time() - start_time:.2f} seconds.")

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
            # Get neighbors for row i directly from CSR structure
            start = P_sym.indptr[i]
            end = P_sym.indptr[i + 1]
            neighbors = set(P_sym.indices[start:end])
            adj_sets[i] = neighbors
        logging.info("Adjacency sets built in %.2f seconds.", time.time() - start_time)
        return adj_sets
    
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
    def load_precomputed_negative_edges(cls, file_path: str) -> list:
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
    
