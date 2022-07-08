import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
import numpy as np
import emcee
from IFMR_tools import pairwise, create_IFMR
from mcmc_functions import logpost_DWDs
from DWD_sets import bad_DWDs_220701 as dont_use_DWDs
from DWD_class import load_DWDs

N_CPU = 10
Nwalkers, Nstep = 1000, 5000
f_MCMC_out = "IFMR_MCMC_outliers"
ifmr_x = np.array([0.5, 2, 4, 8])
ifmr_y = np.array([0.15, 0.6, 0.85, 1.4])
IFMR_true = create_IFMR(ifmr_x, ifmr_y)

###########################################################################
# MCMC starts here

def run_MCMC(DWDs):
    print(len(DWDs))

    Mf_ranges = np.linspace(0, 1.4, len(ifmr_x)+1)
    pos0 = np.array([
        np.random.beta(1.0, 1.0, Nwalkers), #P_weird
        np.random.rayleigh(5.0, Nwalkers), #sig_weird
        np.random.rayleigh(0.02, Nwalkers), #Teff_err
        np.random.rayleigh(0.02, Nwalkers), #logg_err
    ] + [np.random.uniform(*x01, Nwalkers) for x01 in pairwise(Mf_ranges)]).T
    #] + [np.random.uniform(0, Mi, Nwalkers) for Mi in ifmr_x]).T
    Ndim = pos0.shape[1]

    ##Run MCMC
    with Pool(N_CPU) as pool:
        sampler = emcee.EnsembleSampler(Nwalkers, Ndim, logpost_DWDs, \
            args=(DWDs, ifmr_x, True), pool=pool)
        sampler.run_mcmc(pos0, Nstep, progress=True)

    np.save(f"MCMC_output/{f_MCMC_out}_chain.npy", sampler.chain)
    np.save(f"MCMC_output/{f_MCMC_out}_lnprob.npy", sampler.lnprobability)

if __name__ == "__main__":
    DWDs = load_DWDs(exclude_set=dont_use_DWDs)
    run_MCMC(DWDs)
