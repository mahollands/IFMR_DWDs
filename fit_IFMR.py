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
from IFMR_config import N_CPU, Nwalkers, Nstep, f_MCMC_out, \
    f_continue_from, S_T, S_g, ifmr_x

###########################################################################

if f_continue_from:
    ifmr_x, chain, _ = load_fitted_IFMR(f_continue_from)
    pos0 = chain[:,-1,:]
else:
    ifmr_x = np.array(ifmr_x)
    ifmr_y_ = (1.4-0.4)/(8-0.5) * (ifmr_x-0.5) + 0.4
    pos0 = np.array([
        np.random.random(Nwalkers), #P_outlier
        np.random.rayleigh(1.0, Nwalkers), #sigma_outlier
        np.sqrt(invgamma.rvs(a=1, scale=S_T/2, size=Nwalkers)), #Teff_err
        np.sqrt(invgamma.rvs(a=1, scale=S_g/2, size=Nwalkers)), #logg_err
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
