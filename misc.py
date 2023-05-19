from itertools import tee
import numpy as np

def is_sorted(arr):
    return np.all(arr[1:] >= arr[:-1])

def pairwise(iterable):
    """
    Roughly recreates the python 3.10 itertools.pairwise.
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

