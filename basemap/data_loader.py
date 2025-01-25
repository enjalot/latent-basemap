import numpy as np
import os
import glob

class MemmapArrayConcatenator:
    def __init__(self, directories, input_dim):
        self.input_dim = input_dim
        
        # Collect all .npy files from directories
        self.memmap_files = []
        for directory in directories:
            print("directory", directory)
            npy_files = glob.glob(os.path.join(directory, "*.npy"))
            print("npy_files", npy_files)
            self.memmap_files.extend(npy_files)
            
        if not self.memmap_files:
            raise ValueError("No .npy files found in the provided directories")
            
        # Calculate shapes from file sizes
        self.shapes = []
        self.memmaps = []
        
        for path in self.memmap_files:
            try:
                # Get file size and calculate number of samples
                file_size = os.path.getsize(path)
                num_samples = file_size // (4 * input_dim)  # 4 bytes per float32
                shape = (num_samples, input_dim)
                self.shapes.append(shape)
                
                # Load memmap
                memmap = np.memmap(path, dtype='float32', mode='r', shape=shape)
                self.memmaps.append(memmap)
            except Exception as e:
                raise IOError(f"Failed to load memmap file {path}: {str(e)}")
        
        self.total_samples = sum(shape[0] for shape in self.shapes)
        self.feature_dim = input_dim
        self.cumulative_sizes = np.cumsum([0] + [shape[0] for shape in self.shapes])
    
        
    def __array__(self):
        # This allows numpy to treat our object as an array
        return np.concatenate(self.memmaps, axis=0)
    
    @property
    def shape(self):
        return (self.total_samples, self.feature_dim)
    
    @property
    def dtype(self):
        return np.float32
    
    def astype(self, dtype):
        # ParametricUMAP calls this method
        if dtype == np.float32:
            return self
        raise NotImplementedError("Only float32 is supported")

