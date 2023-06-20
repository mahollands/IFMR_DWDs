import os
import ast
from itertools import tee
import numpy as np

MCMC_DIR = "MCMC_output"
METAFILE = "MCMC_meta.dat"

def is_sorted(arr):
    return np.all(arr[1:] >= arr[:-1])

def pairwise(iterable):
    """
    Roughly recreates the python 3.10 itertools.pairwise.
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def load_fitted_IFMR(fname):
    fname = fname.removeprefix(f"{MCMC_DIR}/")
    fname = fname.removesuffix("_chain.npy")
    fname = fname.removesuffix("_lnprob.npy")

    chain = np.load(f"{MCMC_DIR}/{fname}_chain.npy")
    lnp = np.load(f"{MCMC_DIR}/{fname}_lnprob.npy")
    
    if not os.path.exists(METAFILE):
        ifmr_x = np.array([0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8])
        return ifmr_x, chain, lnp

    with open(METAFILE) as F:
        for line in F:
            fname_chain, ifmr_x_str = line.split(" : ")
            if fname == fname_chain:
                ifmr_x = np.array(ast.literal_eval(ifmr_x_str))
                break
        else:
            raise ValueError(f"Could not find meta data for {fname}")
    return ifmr_x, chain, lnp

def write_fitted_IFMR(fname, ifmr_x, sampler):
    np.save(f"{MCMC_DIR}/{fname}_chain.npy", sampler.chain)
    np.save(f"{MCMC_DIR}/{fname}_lnprob.npy", sampler.lnprobability)
    with open("MCMC_meta.dat", 'a') as F:
        F.write(f"{fname} : {list(ifmr_x)}\n")
