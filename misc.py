def pairwise(collection):
    """
    Roughly recreates the python 3.10 itertools.pairwise. Will only work for
    collections, not iterators.
    """
    return zip(collection, collection[1:])

