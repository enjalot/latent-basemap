import numpy as np
from typing import List, Optional

class LanceDBLoader:
    """
    LanceDBLoader provides an interface to interact with a LanceDB table
    as if it were a memmapped array. Data can be retrieved by indexing,
    and a nearest neighbor search is exposed through the `search` method,
    which leverages the existing index in LanceDB.
    """
    
    def __init__(self, db_name: str, table_name: str, columns: Optional[List[str]] = None):
        """
        Initialize the LanceDBLoader by establishing a connection to the LanceDB 
        and opening the designated table.

        Parameters
        ----------
        db_name : str
            Name of the LanceDB database (used to construct the path `/lancedb/{db_name}`).
        scope : str
            The scope or table name to open within the database.
        columns : Optional[List[str]], optional
            List of columns to load when fetching data. If None, all columns are loaded.
        """
        import lancedb
        # Cache the connection and table
        self.db = lancedb.connect(f"{db_name}")
        self.table = self.db.open_table(table_name)
        # Cache the underlying Lance table to expedite take and search operations.
        self.lance_table = self.table.to_lance()
        self.columns = columns

    def __getitem__(self, idx):
        """
        Retrieve rows by integer index, slice, or list/array of indices.

        Parameters
        ----------
        idx : int, slice, list, or np.ndarray
            Indices of rows to retrieve.

        Returns
        -------
        np.ndarray
            The requested row(s) as a NumPy array.
        """
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            indices = list(range(start, stop, step))
            data = np.stack(self.lance_table.take(indices=indices, columns=self.columns)[0].to_numpy())
            return data
        elif isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError("Index out of range")
            data = np.stack(self.lance_table.take(indices=[idx], columns=self.columns)[0].to_numpy())
            # Return a single row
            return data[0]
        elif isinstance(idx, (list, np.ndarray)):
            data = np.stack(self.lance_table.take(indices=idx, columns=self.columns)[0].to_numpy())
            return data
        else:
            raise TypeError("Invalid index type. Must be int, slice, list, or numpy array.")

    def __len__(self):
        """
        Return the total number of rows in the table.

        Returns
        -------
        int
            Total row count.
        """
        try:
            # Use a dedicated method if available.
            return self.table.count_rows()
        except AttributeError:
            # Fallback to using len() on the underlying Lance table.
            return len(self.lance_table)

    @property
    def shape(self):
        """
        Return the shape of the dataset as (n_samples, n_features).

        Returns
        -------
        tuple
            A tuple of (number of rows, number of features).
        """
        # Get first row to determine feature dimension
        first_row = np.stack(self.lance_table.take(indices=[0], columns=self.columns)[0].to_numpy())
        return (len(self), first_row.shape[1])

    def __array__(self, dtype=None, copy=False):
        """
        Allow conversion to a NumPy array using numpy.asarray, exposing
        only the "vector" column from the underlying PyArrow Lance table.
        
        Parameters
        ----------
        dtype : data-type, optional
            If provided, the resulting array is cast to this type.
        copy : bool, optional
            Whether to return a copy of the data.
        
        Returns
        -------
        np.ndarray
            The "vector" column data as a NumPy array, ideally zero-copied.
        """
        print("SHAPE", self.shape)
        data = self[list(range(self.shape[0]))]
        print("DATA", data.shape)
        # Handle dtype conversion if requested.
        if dtype is not None and data.dtype != dtype:
            data = data.astype(dtype, copy=copy)
        elif copy:
            data = data.copy()
        
        print("NP VECTOR", data.shape)
        return data

    @property
    def dtype(self):
        return np.float32
    
    def astype(self, dtype):
        # ParametricUMAP calls this method
        if dtype == np.float32:
            return self
        raise NotImplementedError("Only float32 is supported")

    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        """
        Perform a nearest neighbor search using LanceDB's existing index.

        Parameters
        ----------
        query : np.ndarray
            A single query vector or a batch of query vectors.
        k : int
            Number of nearest neighbors to retrieve.

        Returns
        -------
        np.ndarray
            The search results as a NumPy array.
        """
        # Leverage LanceDB's built-in search functionality.
        result = self.table.search(query).metric("cosine").limit(k).select(["index"]).to_list()
        idx, dists = [], []
        for r in result:
            idx.append(r["index"])
            dists.append(r["_distance"])
        return idx, dists
