import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
import numpy as np
import emcee
from mcmc_functions import logpost_DWD
from DWD_class import load_DWDs

N_CPU = 10
Nwalkers, Nstep = 1000, 1000
f_MCMC_out = "IFMR_MCMC_single"
ifmr_x = np.array([0.0, 8])
ifmr_y_ = [0.5, 1.4]

###########################################################################
# MCMC starts here

def run_MCMC(DWD):
    pos0 = np.array([
        np.random.normal(5.0, 0.1, Nwalkers),
        np.random.normal(1.5, 0.1, Nwalkers),
        ] + [np.random.normal(mf, 0.01, Nwalkers) for mf in ifmr_y_] #IFMR
    ).T
    Ndim = pos0.shape[1]

    ##Run MCMC
    with Pool(N_CPU) as pool:
        sampler = emcee.EnsembleSampler(Nwalkers, Ndim, logpost_DWD, \
            args=(DWD, ifmr_x), pool=pool)
        sampler.run_mcmc(pos0, Nstep, progress=True)
    #sampler = emcee.EnsembleSampler(Nwalkers, Ndim, logpost_DWD, args=(DWD, ifmr_x))
    #sampler.run_mcmc(pos0, Nstep, progress=True)

    np.save(f"MCMC_output/{f_MCMC_out}_chain.npy", sampler.chain)
    np.save(f"MCMC_output/{f_MCMC_out}_lnprob.npy", sampler.lnprobability)

if __name__ == "__main__":
    DWD = load_DWDs(use_set={"runC_WDJ0215+1821AB"})[0]
    print(DWD.vecMtau)
    print(np.sqrt(np.diag(DWD.covMtau_systematics(0.01, 0.01))))
    exit()
    run_MCMC(DWD)
