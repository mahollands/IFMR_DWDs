import os
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
from multiprocessing import Pool
import numpy as np
import emcee
from misc import Taylor_Expand_DWD, pairwise
from mcmc_functions import logpost_DWDs

N_CPU = 10
Nwalkers, Nstep = 100, 2000
ifmr_x = np.array([0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 8])

use_DWDs = {
    "chain_runA_WDJ0855-2637AB.npy",
    "chain_runA_WDJ1015+0806AB.npy", #!
    "chain_runA_WDJ1019+1217AB.npy",
    "chain_runA_WDJ1124-1234AB.npy",
    "chain_runA_WDJ1254-0218AB.npy",
    "chain_runA_WDJ1336-1620AB.npy",
    "chain_runA_WDJ1346-4630AB.npy", #?
    "chain_runA_WDJ1636+0927AB.npy",
    "chain_runA_WDJ1827+0403AB.npy",
    "chain_runA_WDJ1856+2916AB.npy",
    "chain_runA_WDJ1953-1019AB.npy", #?
    "chain_runA_WDJ2131-3459AB.npy",
    "chain_runA_WDJ2142+1329AB.npy", #!
    "chain_runB_WDJ1215+0948AB.npy",
    "chain_runB_WDJ1313+2030AB.npy",
    "chain_runB_WDJ1338+0439AB.npy", #!
    "chain_runB_WDJ1339-5449AB.npy",
    "chain_runB_WDJ1535+2125AB.npy", #?
    "chain_runB_WDJ1729+2916AB.npy", #!
    "chain_runB_WDJ1831-6608AB.npy", #?
    "chain_runB_WDJ1859-5529AB.npy", #?
    "chain_runB_WDJ1904-1946AB.npy", #?
    "chain_runC_WDJ0002+0733AB.npy", #?
    "chain_runC_WDJ0052+1353AB.npy",
    "chain_runC_WDJ0215+1821AB.npy",
    "chain_runC_WDJ0410-1641AB.npy",
    "chain_runC_WDJ0510+0438AB.npy",
    "chain_runC_WDJ2058+1037AB.npy",
    "chain_runC_WDJ2242+1250AB.npy",
    "chain_runD_WDJ0920-4127AB.npy",
    "chain_runD_WDJ1014+0305AB.npy",
    "chain_runD_WDJ1100-1600AB.npy",
    "chain_runD_WDJ1336-1620AB.npy",
    "chain_runD_WDJ1445-1459AB.npy",
    "chain_runD_WDJ1804-6617AB.npy",
    "chain_runD_WDJ1929-3000AB.npy",
    "chain_runD_WDJ2007-3701AB.npy",
    "chain_runD_WDJ2026-5020AB.npy",
    "chain_runD_WDJ2100-6011AB.npy",
    "chain_runD_WDJ2150-6218AB.npy",
}

###########################################################################
# MCMC starts here

def run_MCMC(DWDs):
    print(len(DWDs))

    Mf_ranges = np.linspace(0, 1.4, len(ifmr_x)+1)
    pos0 = np.array([
        np.random.beta(0.5, 0.5, Nwalkers), #P_weird
        np.random.normal(10.0, 1.0, Nwalkers)**2, #V_weird
        np.random.uniform(0.01, 0.05, Nwalkers), #Teff_err
        np.random.uniform(0.01, 0.05, Nwalkers), #logg_err
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
    with open("DWDs_Teffs_loggs.dat", 'rb') as F:
        DWDs = pickle.load(F)
    DWDs = {name : Taylor_Expand_DWD(DWD) for name, DWD in DWDs.items() \
        if name in use_DWDs}
    run_MCMC(DWDs)
