import torch
import numpy as np

class DataPrefetcher:
    """
    Asynchronously prefetch batches from the edge loader using pinned memory for
    overlapping data copy with computation.

    The loader yields (edge_batch, labels) tuples where:
      - edge_batch is a list of (src, dst) index pairs
      - labels is a numpy array of floats (1.0 for positive, 0.0 for negative)

    'dataset' supports integer/list indexing and returns torch Tensors.
    """
    def __init__(self, loader, dataset, device):
        self.loader = iter(loader)
        self.dataset = dataset
        self.device = device
        self.use_cuda = torch.cuda.is_available() and 'cuda' in str(device)
        self.stream = torch.cuda.Stream(device) if self.use_cuda else None
        self.prefetched = None
        self._preload()

    def _preload(self):
        try:
            edge_batch, labels = next(self.loader)
        except StopIteration:
            self.prefetched = None
            return

        src_indexes = [i for i, j in edge_batch]
        dst_indexes = [j for i, j in edge_batch]
        src_values = self.dataset[src_indexes]
        dst_values = self.dataset[dst_indexes]
        targets = torch.as_tensor(labels, dtype=torch.float32)

        # Move to device, using async copy with pinned memory when data is on CPU + CUDA
        data_on_cpu = not src_values.is_cuda

        if self.use_cuda and data_on_cpu:
            with torch.cuda.stream(self.stream):
                src_values = src_values.pin_memory().to(self.device, non_blocking=True)
                dst_values = dst_values.pin_memory().to(self.device, non_blocking=True)
                targets = targets.pin_memory().to(self.device, non_blocking=True)
        elif self.use_cuda:
            # Data already on GPU, just ensure targets are there too
            targets = targets.to(self.device)
        else:
            # CPU or MPS — just move to device (no pin_memory, no async streams)
            src_values = src_values.to(self.device)
            dst_values = dst_values.to(self.device)
            targets = targets.to(self.device)

        self.prefetched = (src_values, dst_values, targets)

    def next(self):
        if self.prefetched is None:
            return None
        if self.stream:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.prefetched
        self._preload()
        return batch
