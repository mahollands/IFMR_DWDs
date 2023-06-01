#!/usr/bin/env python
import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
import numpy as np
import emcee
from IFMR_stats import logpost_DWDs
from DWD_sets import bad_DWDs_230531 as dont_use_DWDs
from DWD_class import load_DWDs
from scipy.stats import invgamma
from misc import load_fitted_IFMR, write_fitted_IFMR

N_CPU = 10
Nwalkers, Nstep = 1000, 10000
f_MCMC_out = "IFMR_MCMC_230601"
ifmr_x = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8])
ifmr_y_ = (1.4-0.4)/(8-0.5) * (ifmr_x-0.5) + 0.4

continue_run = False
f_continue_from = ""

###########################################################################

if continue_run:
    ifmr_x, chain, _ = load_fitted_IFMR(f_continue_from)
    pos0 = chain[:,-1,:]
else:
    pos0 = np.array([
        np.random.random(Nwalkers), #P_outlier
        np.random.rayleigh(1.0, Nwalkers), #sigma_outlier
        np.sqrt(invgamma.rvs(a=1, scale=1.577E-4/2, size=Nwalkers)), #Teff_err
        np.sqrt(invgamma.rvs(a=1, scale=2.978E-3/2, size=Nwalkers)), #logg_err
    ] + [np.random.normal(mf, 0.01, Nwalkers) for mf in ifmr_y_] #IFMR
    ).T
Ndim = pos0.shape[1]

def run_MCMC(DWDs, pos0):
    print(f_MCMC_out)
    print(f"Fitting IFMR with {len(DWDs)} DWDs")

    ##Run MCMC
    with Pool(N_CPU) as pool:
        sampler = emcee.EnsembleSampler(Nwalkers, Ndim, logpost_DWDs, \
            args=(DWDs, ifmr_x, True), pool=pool)
        sampler.run_mcmc(pos0, Nstep, progress=True)

    write_fitted_IFMR(f_MCMC_out, ifmr_x, sampler)

if __name__ == "__main__":
    DWDs = load_DWDs(exclude_set=dont_use_DWDs)
    run_MCMC(DWDs, pos0)
