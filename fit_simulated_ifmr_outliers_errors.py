import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
import numpy as np
import emcee
from IFMR_tools import pairwise
from mcmc_functions import logpost_DWDs
import pickle

N_CPU = 10
Nwalkers, Nstep = 100, 500
f_MCMC_out = "IFMR_MCMC_simulated_outliers"
ifmr_x = np.array([0.5, 2, 4, 8])

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
    with open("DWDs_simulated.pkl", 'rb') as F:
        DWDs = pickle.load(F)
    run_MCMC(DWDs)
