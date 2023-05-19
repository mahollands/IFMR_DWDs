import ast
from itertools import tee
import numpy as np

MCMC_DIR = "MCMC_output"

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
    with open("MCMC_meta.dat") as F:
        for line in F:
            fname_chain, ifmr_x_str = line.split(" : ")
            if fname == fname_chain:
                ifmr_x = np.array(ast.literal_eval(ifmr_x_str))
                break
        else:
            raise ValueError(f"Could not find meta data for {fname}")
    chain = np.load(f"{MCMC_DIR}/{fname}_chain.npy")
    lnp = np.load(f"{MCMC_DIR}/{fname}_lnprob.npy")
    return ifmr_x, chain, lnp

def write_fitted_IFMR(fname, ifmr_x, sampler):
    np.save(f"{MCMC_DIR}/{fname}_chain.npy", sampler.chain)
    np.save(f"{MCMC_DIR}/{fname}_lnprob.npy", sampler.lnprobability)
    with open("MCMC_meta.dat", 'a') as F:
        F.write(f"{fname} : {list(ifmr_x)}\n")
