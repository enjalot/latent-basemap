import torch
import numpy as np

class DataPrefetcher:
    """
    Asynchronously prefetch batches from the edge loader using pinned memory for
    overlapping data copy with computation.
    
    This wrapper assumes:
      - 'loader' yields edge_batch lists,
      - 'dataset' and 'target_dataset' support indexing and return NumPy arrays.
    """
    def __init__(self, loader, dataset, target_dataset, device):
        self.loader = iter(loader)
        self.dataset = dataset
        self.target_dataset = target_dataset
        self.device = device
        self.stream = torch.cuda.Stream(device) if torch.cuda.is_available() else None
        self.prefetched = None
        self._preload()
        
    def _preload(self):
        try:
            edge_batch = next(self.loader)
        except StopIteration:
            self.prefetched = None
            return
        
        # If a single edge is returned instead of a batch (i.e. a tuple of 2 ints),
        # then wrap it in a list so that subsequent code can always iterate over a list.
        if (len(edge_batch) == 2 and
            isinstance(edge_batch[0], (np.integer, int))):
            edge_batch = [edge_batch]
        
        # Get indexes and retrieve raw data
        try:
            src_indexes = [i for i, j in edge_batch]
        except:
            print(f"Error: {edge_batch}")
        dst_indexes = [j for i, j in edge_batch]
        src_values = self.dataset[src_indexes]
        dst_values = self.dataset[dst_indexes]
        targets   = self.target_dataset[edge_batch]

        # Use a dedicated CUDA stream for asynchronous copy if available.
        if self.stream:
            with torch.cuda.stream(self.stream):
                src_values = torch.as_tensor(src_values, dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
                dst_values = torch.as_tensor(dst_values, dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
                targets   = torch.as_tensor(targets,   dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
        else:
            src_values = torch.as_tensor(src_values, dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
            dst_values = torch.as_tensor(dst_values, dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
            targets   = torch.as_tensor(targets,   dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
            
        self.prefetched = (src_values, dst_values, targets)
        
    def next(self):
        if self.prefetched is None:
            return None
        # Wait for the previous async copy to finish
        if self.stream:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.prefetched
        self._preload()
        return batch 