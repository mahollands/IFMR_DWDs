import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
import numpy as np
import emcee
from misc import pairwise
from mcmc_functions import logpost_DWDs
from DWD_sets import good_DWDs_220630 as use_DWDs
from DWD_class import load_DWDs

N_CPU = 10
Nwalkers, Nstep = 500, 100
f_MCMC_out = "IFMR_MCMC_outliers"
ifmr_x = np.array([0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 8])
#ifmr_x = np.array([0, 2, 4, 8])

###########################################################################
# MCMC starts here

def run_MCMC(DWDs):
    print(len(DWDs))

    Mf_ranges = np.linspace(0, 1.4, len(ifmr_x)+1)
    pos0 = np.array([
        np.random.uniform(0.01, 0.05, Nwalkers), #Teff_err
        np.random.uniform(0.01, 0.05, Nwalkers), #logg_err
    ] + [np.random.uniform(*x01, Nwalkers) for x01 in pairwise(Mf_ranges)]).T
    Ndim = pos0.shape[1]

    #Run MCMC
    with Pool(N_CPU) as pool:
        sampler = emcee.EnsembleSampler(Nwalkers, Ndim, logpost_DWDs, \
            args=(DWDs, ifmr_x), pool=pool)
        sampler.run_mcmc(pos0, Nstep, progress=True)

    np.save(f"MCMC_output/{f_MCMC_out}_chain.npy", sampler.chain)
    np.save(f"MCMC_output/{f_MCMC_out}_lnprob.npy", sampler.lnprobability)


if __name__ == "__main__":
    DWDs = load_DWDs(use_set=use_DWDs)
    run_MCMC(DWDs)
