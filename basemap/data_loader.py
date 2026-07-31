import numpy as np
import os
import glob
import logging
import hashlib

from .artifact_identity import canonical_json


def prefix_l2_preprocessing_contract(*, source_dimension: int,
                                     output_dimension: int,
                                     normalize: bool) -> tuple[dict, dict]:
    """Return the canonical preprocessing body and flat runtime stamp."""
    if not isinstance(source_dimension, int) or source_dimension <= 0:
        raise ValueError("source_dimension must be a positive integer")
    if (not isinstance(output_dimension, int) or output_dimension <= 0 or
            output_dimension > source_dimension):
        raise ValueError("output_dimension must be in [1, source_dimension]")
    if normalize is not True and normalize is not False:
        raise ValueError("normalize must be an exact boolean")
    if output_dimension < source_dimension and not normalize:
        raise ValueError("a reduced prefix must be L2-renormalized explicitly")
    schema = "prefix-l2-input-preprocessing-v1"
    operation = (
        "identity-fp32-cast"
        if not normalize
        else "first-dimensions-then-row-l2-normalize-fp32"
    )
    body = {
        "schema": schema,
        "operation": operation,
        "source_dimension": int(source_dimension),
        "effective_dimension": int(output_dimension),
        "slice_start": 0,
        "slice_stop": int(output_dimension),
        "l2_renormalized": bool(normalize),
        "compute_dtype": "<f4",
        "zero_or_nonfinite_policy": (
            "reject" if normalize else "preserve-for-downstream-guards"
        ),
    }
    digest = hashlib.sha256(canonical_json(body)).hexdigest()
    receipt = {**body, "identity_sha256": digest}
    stamp = {
        "input_preprocessing_schema": schema,
        "input_preprocessing_operation": operation,
        "input_source_dimension": int(source_dimension),
        "input_effective_dimension": int(output_dimension),
        "input_slice_start": 0,
        "input_slice_stop": int(output_dimension),
        "input_l2_renormalized": bool(normalize),
        "input_preprocessing_compute_dtype": "<f4",
        "input_preprocessing_sha256": digest,
    }
    return receipt, stamp


class PrefixL2NormalizedArray:
    """Lazy, receipt-bearing prefix projection of a two-dimensional array.

    Matryoshka embeddings are useful to the projector only if the feature
    prefix is selected *before* device residency and, for a reduced prefix,
    L2-normalized row by row.  This wrapper keeps the source memmapped, applies
    that operation to each requested row block, and exposes a flat execution
    stamp that the trainer records before update zero.

    ``normalize=False`` is the exact full-dimensional control: it casts the
    source values to float32 but does not perturb them.  ``normalize=True``
    fails closed on zero or non-finite rows instead of manufacturing values.
    """

    SCHEMA = "prefix-l2-input-preprocessing-v1"

    def __init__(self, source, *, source_dimension: int, output_dimension: int,
                 normalize: bool, source_paths=None):
        receipt, stamp = prefix_l2_preprocessing_contract(
            source_dimension=source_dimension,
            output_dimension=output_dimension,
            normalize=normalize)
        shape = getattr(source, "shape", None)
        if (not isinstance(shape, tuple) or len(shape) != 2 or
                int(shape[1]) != source_dimension):
            raise ValueError(
                f"source shape {shape!r} does not match dimension "
                f"{source_dimension}")
        self.source = source
        self.source_dimension = int(source_dimension)
        self.output_dimension = int(output_dimension)
        self.normalize = bool(normalize)
        self.shape = (int(shape[0]), self.output_dimension)
        self.dtype = np.dtype("float32")
        paths = source_paths
        if paths is None:
            paths = (getattr(source, "loaded_shard_paths", None) or
                     getattr(source, "shard_paths", None) or [])
            filename = getattr(source, "filename", None)
            if not paths and isinstance(filename, (str, os.PathLike)):
                paths = [filename]
        self.loaded_shard_paths = [os.path.realpath(os.fspath(p)) for p in paths]
        self.shard_paths = list(self.loaded_shard_paths)

        self.preprocessing = receipt
        self.execution_preprocessing_stamp = stamp

    def __len__(self):
        return self.shape[0]

    def astype(self, dtype):
        if np.dtype(dtype) == np.dtype("float32"):
            return self
        raise NotImplementedError("PrefixL2NormalizedArray exposes float32 only")

    def __array__(self, dtype=None, copy=None):
        raise RuntimeError(
            "refuse to materialize the full lazy preprocessed input")

    def _preprocess(self, values):
        array = np.asarray(values, dtype=np.float32)
        result = np.array(
            array[..., :self.output_dimension], dtype=np.float32, copy=True)
        if self.normalize:
            if not np.isfinite(result).all():
                raise ValueError("input prefix contains non-finite values")
            if result.ndim == 1:
                norm = float(np.linalg.norm(result))
                if not np.isfinite(norm) or norm <= 0.0:
                    raise ValueError("input prefix has zero/non-finite L2 norm")
                result /= np.float32(norm)
            else:
                norms = np.linalg.norm(result, axis=-1).astype(
                    np.float32, copy=False)
                if (not np.isfinite(norms).all() or
                        np.any(norms <= np.float32(0.0))):
                    raise ValueError(
                        "input prefix has zero/non-finite L2 norm")
                result /= norms[..., None]
        return result

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("expected a two-dimensional index")
            rows, columns = key
            return self.__getitem__(rows)[..., columns]
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if isinstance(self.source, np.ndarray):
                return self._preprocess(self.source[slice(start, stop, step)])
            # Array-style row selection avoids the legacy memmap loader's
            # scalar loop and keeps million-row device uploads tractable.
            key = np.arange(start, stop, step, dtype=np.int64)
        return self._preprocess(self.source[key])

class MemmapArrayConcatenator:
    def __init__(self, directories, input_dim, testing: bool = False):
        self.input_dim = input_dim
        
        # Collect all .npy files from directories. SORTED per directory so the
        # concatenation order (hence row indices, and the graph/data pairing) is
        # DETERMINISTIC — glob order is arbitrary (S0).
        self.memmap_files = []
        for directory in directories:
            if testing:
                npy_files = glob.glob(os.path.join(directory, "data-00000-*.npy"))
            else:
                npy_files = glob.glob(os.path.join(directory, "*.npy"))
            self.memmap_files.extend(sorted(npy_files))

        if not self.memmap_files:
            raise ValueError("No .npy files found in the provided directories")

        # Load standard .npy files through NumPy so dtype/header metadata is
        # respected. Most local embedding shards are float16 .npy arrays, not
        # raw float32 buffers.
        self.shapes = []
        self.memmaps = []
        self.loaded_shard_paths = []   # S0: ordered paths of ACCEPTED shards only

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
                self.loaded_shard_paths.append(path)   # S0: only accepted shards, in order
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
