#!/usr/bin/env python
import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
import numpy as np
import emcee
from IFMR_stats import logpost_DWDs
from DWD_sets import bad_DWDs_230511 as dont_use_DWDs
from DWD_class import load_DWDs
from scipy.stats import invgamma

N_CPU = 10
Nwalkers, Nstep = 1000, 2000
f_MCMC_out = "IFMR_MCMC_outliers_230511_extend01"
ifmr_x = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8])
ifmr_y_ = (1.4-0.4)/(8-0.5) * (ifmr_x-0.5) + 0.4

continue_run = True
continue_from = "IFMR_MCMC_outliers_230511"

###########################################################################
# MCMC starts here

def run_MCMC(DWDs):
    print(len(DWDs))

    if continue_run:
        pos0 = np.load(f"MCMC_output/{continue_from}_chain.npy")[:,-1,:]
    else:
        pos0 = np.array([
            np.random.random(Nwalkers), #P_weird
            np.random.rayleigh(1.0, Nwalkers), #sig_weird
            np.sqrt(invgamma.rvs(a=1, scale=1.577E-4/2, size=Nwalkers)), #Teff_err
            np.sqrt(invgamma.rvs(a=1, scale=2.978E-3/2, size=Nwalkers)), #logg_err
        ] + [np.random.normal(mf, 0.01, Nwalkers) for mf in ifmr_y_] #IFMR
        ).T

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
    with open("MCMC_meta.dat", 'a') as F:
        F.write(f"{f_MCMC_out} : {list(ifmr_x)}\n")
