import pickle
from math import log
from multiprocessing import Pool
import numpy as np
import numba
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats
import emcee

N_CPU = 10
Nwalkers, Nstep, Nmarginal = 200, 500, 10_000
ifmr_x = np.array([0, 8])
ifmr_x = np.array([0, 2, 4, 8])
ifmr_x = np.array([0, 1, 2, 3, 4, 6, 8])

use_DWDs = {
    "chain_runA_WDJ0855-2637AB.npy",
    "chain_runA_WDJ1015+0806AB.npy",
    "chain_runA_WDJ1019+1217AB.npy",
    "chain_runA_WDJ1124-1234AB.npy",
    "chain_runA_WDJ1254-0218AB.npy",
    "chain_runA_WDJ1336-1620AB.npy",
    #"chain_runA_WDJ1346-4630AB.npy", #?
    #"chain_runA_WDJ1445+2921AB.npy", #?
    "chain_runA_WDJ1636+0927AB.npy",
    "chain_runA_WDJ1856+2916AB.npy",
    "chain_runA_WDJ1907+0136AB.npy",
    #"chain_runA_WDJ1953-1019AB.npy", #?
    "chain_runA_WDJ2131-3459AB.npy",
    "chain_runA_WDJ2142+1329AB.npy",
    #"chain_runA_WDJ2223+2201AB.npy", #?
    "chain_runB_WDJ1313+2030AB.npy",
    "chain_runB_WDJ1338+0439AB.npy",
    "chain_runB_WDJ1339-5449AB.npy",
    #"chain_runB_WDJ1535+2125AB.npy", #?
    "chain_runB_WDJ1729+2916AB.npy",
    #"chain_runB_WDJ1831-6608AB.npy", #?
    "chain_runB_WDJ1859-5529AB.npy",
    #"chain_runB_WDJ1904-1946AB.npy", #?
    #"chain_runC_WDJ0007-1605AB.npy",
    "chain_runC_WDJ0052+1353AB.npy",
    #"chain_runC_WDJ0101-1629AB.npy", #?
    #"chain_runC_WDJ0109-1042AB.npy", #?
    "chain_runC_WDJ0120-1622AB.npy",
    "chain_runC_WDJ0215+1821AB.npy",
    "chain_runC_WDJ0410-1641AB.npy",
    "chain_runC_WDJ0510+0438AB.npy",
    "chain_runC_WDJ2058+1037AB.npy",
    "chain_runC_WDJ2122+3005AB.npy",
    "chain_runC_WDJ2139-1003AB.npy",
    "chain_runC_WDJ2242+1250AB.npy",
}

with open("DWDs.dat", 'rb') as F:
    DWDs = pickle.load(F)
DWDs = {DWD : vals for DWD, vals in DWDs.items() if DWD in use_DWDs}


#MS lifetime from MESA data
M_init, t_pre = np.loadtxt("MESA_lifetime.dat", unpack=True, skiprows=1)
t_pre /= 1e9 #to Gyr
log_tau_fun = interp1d(M_init, np.log10(t_pre), kind='cubic', bounds_error=False)

def MSLT(Mi):
    """
    Main sequence lifetime
    """
    return 10**log_tau_fun(Mi)

def loglike_Mi12(Mi1, Mi2, theta, DWD):
    """
    params are the MS star masses and the y values of the IFMR at values. DWD
    contains a vector of Mf1, Mf2, dtau_cool and the corresponding covariance
    matrix.
    """
    vec, Sigma = DWD
    IFMR = interp1d(ifmr_x, theta)
    Mf1, Mf2 = IFMR([Mi1, Mi2])
    tau1, tau2 = MSLT([Mi1, Mi2])
    X = np.array([Mf1, Mf2, tau2-tau1])
    return stats.multivariate_normal.logpdf(X.T, mean=vec, cov=Sigma) + 100

@numba.vectorize
def logprior_Mi12(Mi1, Mi2):
    """
    priors on inital masses
    """
    if not (0.6 < Mi1 < 8) or not (0.6 < Mi2 < 8):
        return -np.inf
    return -2.35*log(Mi1*Mi2) #Lazy Salpeter IMF as initial mass prior

def logpost_Mi12(Mi1, Mi2, theta, DWD):
    lp = logprior_Mi12(Mi1, Mi2)
    return lp + loglike_Mi12(Mi1, Mi2, theta, DWD)

def grad_IFMR_i(Mf, ifmr_y, ifmr_x):
    """
    The gradient of the inverse IFMR, for a piecewise
    linear and piecewise continuous IFMR. This essentially
    gets the jacobian dMi/dMf
    """
    segments = [Mf < y for y in ifmr_y]
    grads = [(x1-x0)/(y1-y0) for (x0, x1, y0, y1)
        in zip(ifmr_x, ifmr_x[1:], ifmr_y, ifmr_y[1:])]
    grads.insert(0, 0)
    return np.select(segments, grads)

def loglike_DWD(theta, name, DWD):
    """
    Marginal distribution :
    P(theta, DWD) = \iint P(Mi1, Mi2, theta | DWD) dMi1 dMi2
    """
    vec, cov = DWD
    vec, cov = vec[:2], cov[:2,:2]
    Mf12 = np.random.multivariate_normal(vec, cov, Nmarginal)
    #reject samples outside of IFMR
    ok = (Mf12 > theta[0]) & (Mf12 < theta[-1])
    ok = np.all(ok, axis=1)
    Mf12 = Mf12[ok,:]
    log_weights = -stats.multivariate_normal.logpdf(Mf12, mean=vec, cov=cov)
    IFMR_i = interp1d(theta, ifmr_x)
    Mi12 = IFMR_i(Mf12)
    jac1, jac2 = grad_IFMR_i(Mf12, theta, ifmr_x).T
    log_probs = logpost_Mi12(*Mi12.T, theta, DWD)
    integrand = np.exp(log_probs + log_weights) * jac1 * jac2
    I = np.mean(integrand)
    ret = -np.inf if np.isnan(I) or I <= 0 else log(I)
    #print(name, ret)
    #return -np.inf if np.isnan(I) or I <= 0 else log(I)
    return ret

def loglike_DWDs(theta, DWDs):
    """
    log likelihood for theta for all DWDs
    """
    return sum(loglike_DWD(theta, name, DWD) for name, DWD in DWDs.items())

def logprior(theta):
    """
    priors on IFMR y values
    """
    if not all(0 < y < 1.4 for y in theta):
        #piecewise points in IFMR must be valid WD masses
        return -np.inf
    if not list(theta) == sorted(theta):
        #piecewise points in IFMR must be increasing
        return -np.inf
    return 0

def logpost_DWD(theta, DWD):
    """
    Test posterior for fitting a single DWD only
    """
    lp = logprior(theta)
    return lp if lp == -np.inf else lp + loglike_DWD(theta, DWD)

def logpost(theta, DWDs):
    """
    posterior distribution for fitting IFMR to all DWDs
    """
    lp = logprior(theta)
    return lp if lp == -np.inf else lp + loglike_DWDs(theta, DWDs)

###########################################################################
# MCMC starts here

def run_MCMC(DWDs):
    print(len(DWDs))

    pos0 = np.array([
        np.random.uniform(0.0, 0.1, Nwalkers),
        np.random.uniform(0.1, 0.2, Nwalkers),
        np.random.uniform(0.50, 0.60, Nwalkers),
        np.random.uniform(0.60, 0.70, Nwalkers),
        np.random.uniform(0.70, 0.80, Nwalkers),
        np.random.uniform(0.90, 1.00, Nwalkers),
        np.random.uniform(1.2, 1.4, Nwalkers),
    ]).T
    Ndim = pos0.shape[1]

    #Run MCMC
    with Pool(N_CPU) as pool:
        sampler = emcee.EnsembleSampler(Nwalkers, Ndim, logpost, args=(DWDs,), pool=pool)
        sampler.run_mcmc(pos0, Nstep, progress=True)

    np.save("IFMR_MCMC.npy", sampler.chain)
    np.save("IFMR_MCMC_lnprob.npy", sampler.lnprobability)

if __name__ == "__main__":
    run_MCMC(DWDs)
