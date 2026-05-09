import numpy as np
import os
import glob
import logging

class MemmapArrayConcatenator:
    def __init__(self, directories, input_dim, testing: bool = False):
        self.input_dim = input_dim
        
        # Collect all .npy files from directories
        self.memmap_files = []
        for directory in directories:
            if testing:
                npy_files = glob.glob(os.path.join(directory, "data-00000-*.npy"))
            else:
                npy_files = glob.glob(os.path.join(directory, "*.npy"))
            self.memmap_files.extend(npy_files)
            
        if not self.memmap_files:
            raise ValueError("No .npy files found in the provided directories")
            
        # Load standard .npy files through NumPy so dtype/header metadata is
        # respected. Most local embedding shards are float16 .npy arrays, not
        # raw float32 buffers.
        self.shapes = []
        self.memmaps = []
        
        for path in self.memmap_files:
            try:
                memmap = np.load(path, mmap_mode="r")
                if memmap.ndim != 2:
                    logging.debug("Skipping non-2D npy file %s with shape %s", path, memmap.shape)
                    continue
                if memmap.shape[1] != input_dim:
                    logging.debug(
                        "Skipping npy file %s with shape %s; expected dim %d",
                        path, memmap.shape, input_dim
                    )
                    continue
                self.shapes.append(memmap.shape)
                self.memmaps.append(memmap)
            except Exception as e:
                raise IOError(f"Failed to load memmap file {path}: {str(e)}")

        if not self.memmaps:
            raise ValueError(
                f"No 2D .npy files with feature dimension {input_dim} found in "
                f"{directories}"
            )
        
        self.total_samples = sum(shape[0] for shape in self.shapes)
        self.feature_dim = input_dim
        self.cumulative_sizes = np.cumsum([0] + [shape[0] for shape in self.shapes])
    
        
    def __array__(self, dtype=None, copy=False):
        # This allows numpy to treat our object as an array
        result = np.concatenate(self.memmaps, axis=0)
        if dtype is not None:
            result = result.astype(dtype)
        if copy:
            result = result.copy()
        return result
    
    @property
    def shape(self):
        return (self.total_samples, self.feature_dim)
    
    @property
    def dtype(self):
        return np.float32

    def __len__(self):
        return self.total_samples
    
    def astype(self, dtype):
        # ParametricUMAP calls this method
        if dtype == np.float32:
            return self
        raise NotImplementedError("Only float32 is supported")

    def __getitem__(self, idx):
        # Handle vectorized integer indexing without materializing every shard.
        if isinstance(idx, (list, tuple, np.ndarray)):
            idx = np.asarray(idx)
            if idx.ndim != 1:
                raise IndexError("Only 1D index arrays are supported")
            result = np.empty((len(idx), self.feature_dim), dtype=np.float32)
            normalized_idx = idx.astype(np.int64, copy=True)
            normalized_idx[normalized_idx < 0] += self.total_samples
            if np.any((normalized_idx < 0) | (normalized_idx >= self.total_samples)):
                raise IndexError("Index out of bounds")

            array_ids = np.searchsorted(self.cumulative_sizes, normalized_idx, side='right') - 1
            for array_id in np.unique(array_ids):
                out_positions = np.flatnonzero(array_ids == array_id)
                local_idx = normalized_idx[out_positions] - self.cumulative_sizes[array_id]
                result[out_positions] = self.memmaps[array_id][local_idx]
            return result

        # Handle slice objects
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else self.total_samples
            step = idx.step if idx.step is not None else 1
            
            # Create a new MemmapArrayConcatenator with the sliced data
            indices = range(start, stop, step)
            result = np.zeros((len(indices), self.feature_dim), dtype=np.float32)
            
            for i, idx in enumerate(indices):
                result[i] = self[idx]  # Use integer indexing
            return result
            
        # Handle integer indexing
        if idx < 0:
            idx += self.total_samples
        if not 0 <= idx < self.total_samples:
            raise IndexError("Index out of bounds")
            
        # Find which memmap contains this index
        array_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        local_idx = idx - self.cumulative_sizes[array_idx]
        
        return self.memmaps[array_idx][local_idx]
