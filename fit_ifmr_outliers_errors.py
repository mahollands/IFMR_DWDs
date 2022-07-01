import os
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
from multiprocessing import Pool
import numpy as np
import emcee
from misc import Taylor_Expand_DWD, pairwise
from mcmc_functions import logpost_DWDs
from DWD_sets import bad_DWDs_220701 as dont_use_DWDs

N_CPU = 10
Nwalkers, Nstep = 100, 500
ifmr_x = np.array([0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 8])

###########################################################################
# MCMC starts here

def run_MCMC(DWDs):
    print(len(DWDs))

    Mf_ranges = np.linspace(0, 1.4, len(ifmr_x)+1)
    pos0 = np.array([
        np.random.beta(0.5, 0.5, Nwalkers), #P_weird
        np.random.normal(10.0, 1.0, Nwalkers)**2, #V_weird
        np.random.exponential(0.01, Nwalkers), #Teff_err
        np.random.exponential(0.01, Nwalkers), #logg_err
    ] + [np.random.uniform(*x01, Nwalkers) for x01 in pairwise(Mf_ranges)]).T
    Ndim = pos0.shape[1]

    #Run MCMC
    with Pool(N_CPU) as pool:
        sampler = emcee.EnsembleSampler(Nwalkers, Ndim, logpost_DWDs, \
            args=(DWDs, ifmr_x, True), pool=pool)
        sampler.run_mcmc(pos0, Nstep, progress=True)

    np.save("IFMR_MCMC_outliers.npy", sampler.chain)
    np.save("IFMR_MCMC_outliers_lnprob.npy", sampler.lnprobability)

if __name__ == "__main__":
    with open("DWDs_Teffs_loggs.pkl", 'rb') as F:
        DWDs = pickle.load(F)
    DWDs = {name : Taylor_Expand_DWD(DWD) for name, DWD in DWDs.items() \
        #if name not in use_DWDs}
        if name not in dont_use_DWDs}
    run_MCMC(DWDs)
