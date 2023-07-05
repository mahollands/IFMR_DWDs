#!/usr/bin/env python
import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
import numpy as np
import emcee
from IFMR_stats import logpost_DWDs
from DWDs import load_DWDs, bad_DWDs
from scipy.stats import invgamma
from misc import load_fitted_IFMR, write_fitted_IFMR
import IFMR_config as cfg

###########################################################################

def initialise_walkers():
    if cfg.f_continue_from:
        ifmr_x, chain, _ = load_fitted_IFMR(cfg.f_continue_from)
        return ifmr_x, chain[:,-1,:]
    ifmr_x = np.array(cfg.ifmr_x)
    ifmr_y_ = (1.4-0.4)/(8-0.5) * (ifmr_x-0.5) + 0.4
    pos0 = np.array([
        np.random.random(cfg.Nwalkers), #P_outlier
        np.random.rayleigh(1.0, cfg.Nwalkers), #sigma_outlier
        np.sqrt(invgamma.rvs(a=1, scale=cfg.S_T/2, size=cfg.Nwalkers)), #Teff_err
        np.sqrt(invgamma.rvs(a=1, scale=cfg.S_g/2, size=cfg.Nwalkers)), #logg_err
    ] + [np.random.normal(mf, 0.01, cfg.Nwalkers) for mf in ifmr_y_] #IFMR
    ).T
    return ifmr_x, pos0

def run_MCMC(DWDs, ifmr_x, pos0):
    print(cfg.f_MCMC_out)
    print(f"Fitting IFMR with {len(DWDs)} DWDs")

    ##Run MCMC
    Ndim = pos0.shape[1]
    with Pool(cfg.N_CPU) as pool:
        sampler = emcee.EnsembleSampler(cfg.Nwalkers, Ndim, logpost_DWDs, \
            args=(DWDs, ifmr_x, True), pool=pool)
        sampler.run_mcmc(pos0, cfg.Nstep, progress=True)

    write_fitted_IFMR(cfg.f_MCMC_out, ifmr_x, sampler)

if __name__ == "__main__":
    DWDs = load_DWDs(cfg.f_DWDs, exclude_set=bad_DWDs[cfg.exclude_set])
    ifmr_x, pos0 = initialise_walkers()
    run_MCMC(DWDs, ifmr_x, pos0)
