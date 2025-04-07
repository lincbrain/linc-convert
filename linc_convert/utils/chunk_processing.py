import numpy as np

from linc_convert.utils.math import ceildiv


def chunk_slice_generator(arr_shape, chunk_shape):
    """
    Generate slice indices for chunking an array based on its shape and the provided chunk_shape
    for the last dimensions.

    Parameters:
        arr_shape (tuple): The shape of the array.
        chunk_shape (tuple): A tuple of integers representing the chunk size for each of the last dimensions.

    Yields:
        tuple: A tuple (index, full_slice) where:
            - index is a tuple representing the multi-index of the chunk.
            - full_slice is a tuple of slice objects (prefixed by an Ellipsis to preserve any non-chunked dimensions)
              that can be used to index into an array of shape arr_shape.
    """
    # Ensure chunk_shape is a tuple
    chunk_shape = tuple(chunk_shape)

    # Consider only the last dimensions that are meant to be chunked
    shape = arr_shape[-len(chunk_shape):]

    # Calculate the number of chunks along each dimension
    n_chunks = [ceildiv(dim, c) for dim, c in zip(shape, chunk_shape)]

    # Generate all multi-indices for the chunk grid
    for index in np.ndindex(*n_chunks):
        # Create slice objects for each dimension, ensuring we don't go beyond the array's bounds
        slices = tuple(slice(i * c, min((i + 1) * c, dim))
                       for i, c, dim in zip(index, chunk_shape, shape))
        # Prepend an Ellipsis to preserve any preceding (non-chunked) dimensions
        full_slice = (...,) + slices
        yield index, full_slice

