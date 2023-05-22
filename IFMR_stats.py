"""
MCMC routines for IFMR fitting with DWD binaries
"""
from math import log, log1p
from functools import partial
import numpy as np
import numba
from scipy import stats
from scipy.special import logsumexp
from IFMR_tools import IFMR_cls, MSLT
from misc import is_sorted

MONOTONIC_IFMR = True
MONOTONIC_MASS_LOSS = False
MCH_PRIOR = True
STRICT_MASS_LOSS = True
N_MARGINALISE = 10000

##################################
# constants
log_weights_uniform = 2*log(8-0.6)

def loglike_Mi12(Mi12, vec, cov, IFMR, outliers=False, scale_weird=None):
    """
    Computes the likelihood of an IFMR and initial masses for one DWD with
    measured final masses and difference in WD cooling ages (and their
    covariance). This is optionally computed for either the coeval or outlier
    distributions.
    """
    Mf12 = IFMR(Mi12)
    tau1_ms, tau2_ms = MSLT(Mi12)
    dtau_ms = tau1_ms-tau2_ms
    X, cov_ = np.vstack([Mf12, -dtau_ms]), np.copy(cov)
    if outliers:
        cov_[2,2] += scale_weird**2
    return stats.multivariate_normal.logpdf(X.T, mean=vec, cov=cov_)

def loglike_Mi12_mixture(Mi12, vec, cov, IFMR, P_weird, scale_weird, separate=False):
    """
    Computes the likelihood of an IFMR and initial masses for one DWD with
    measured final masses and difference in WD cooling ages (and their
    covariance), and under the assumption of a fraction of systems being
    outliers drawn from a broader dtau_c distribution.
    """
    args = Mi12, vec, cov, IFMR
    logL_coeval  = loglike_Mi12(*args, outliers=False)
    logL_weird = loglike_Mi12(*args, outliers=True, scale_weird=scale_weird)

    if isinstance(logL_coeval, np.ndarray):
        logL_coeval[np.isnan(logL_coeval)] = -np.inf
        logL_weird[np.isnan(logL_weird)] = -np.inf
    else:
        if np.isnan(logL_coeval):
            logL_coeval = -np.inf
        if np.isnan(logL_weird):
            logL_weird = -np.inf

    logL_coeval += log1p(-P_weird)
    logL_weird += log(P_weird)
    if separate:
        return logL_coeval, logL_weird
    return np.logaddexp(logL_coeval, logL_weird)

@numba.vectorize
def logprior_Mi12(Mi1, Mi2):
    """
    Priors on inital masses, i.e.
    P(Mi1, Mi2) = (M1*M2)**-2.3
    """
    if not (0.6 < Mi1 < 8.0 and 0.6 < Mi2 < 8.0):
        return -np.inf
    return -2.3*log(Mi1*Mi2) #Kroupa IMF for m>0.5Msun

def loglike_DWD(params, DWD, IFMR, outliers=False):
    """
    log of marginal distribution:
    P(DWD | IFMR, theta) = \\iint P(Mi1, Mi2, DWD | IFMR, theta) dMi1 dMi2
    """
    if outliers:
        P_weird, scale_weird, Teff_err, logg_err = params
        loglike_Mi12_ = partial(loglike_Mi12_mixture, \
            P_weird=P_weird, scale_weird=scale_weird)
    else:
        Teff_err, logg_err = params
        loglike_Mi12_ = loglike_Mi12
    covMdtau = DWD.covMdtau_systematics(Teff_err, logg_err)
    vecM, covM = DWD.vecMdtau[:2], covMdtau[:2,:2]

    if MONOTONIC_IFMR:
        Mi12, Mf12 = IFMR.draw_Mi_samples(vecM, covM, N_MARGINALISE)
        if len(Mf12) <= 1:
            return -np.inf
        log_weights = -stats.multivariate_normal.logpdf(Mf12, mean=vecM, cov=covM)
        jac1, jac2 = IFMR.inv_grad(Mf12).T
    else:
        Mi12 = np.random.uniform(0.6, 8, (2, N_MARGINALISE))
        log_weights = log_weights_uniform
        jac1, jac2 = 1, 1

    #importance sampling
    log_like = loglike_Mi12_(Mi12, DWD.vecMdtau, covMdtau, IFMR)
    log_integrand = logprior_Mi12(*Mi12) + log_like + log_weights
    return logsumexp(log_integrand, b=np.abs(jac1*jac2)) - log(N_MARGINALISE)

def loglike_DWDs(params, DWDs, IFMR, outliers=False):
    """
    log likelihood for IFMR (and hyperparams) for all DWDs
    """
    return sum(loglike_DWD(params, DWD, IFMR, outliers=outliers) for DWD in DWDs)

def logprior_IFMR(IFMR):
    """
    Prior on the IFMR only. Specifically this amounts to cuts on the 
    """
    if STRICT_MASS_LOSS and not all(0 < q < 1 for q in IFMR.mass_loss):
        return -np.inf

    if MCH_PRIOR and np.any(IFMR.y > 1.4):
        return -np.inf

    if MONOTONIC_IFMR and not is_sorted(IFMR.y):
        return -np.inf

    if MONOTONIC_MASS_LOSS and not is_sorted(IFMR.mass_loss):
        return -np.inf

    return 0

def logprior_DWD(Mi12, IFMR):
    """
    prior for only one DWD
    """
    return logprior_Mi12(*Mi12) + logprior_IFMR(IFMR)

def logprior_DWDs(params, IFMR, outliers=False):
    """
    priors on IFMR all and ifmr related parameters
    """
    if outliers:
        P_weird, scale_weird, Teff_err, logg_err = params
        if not 0 < P_weird < 1 or not 0 < scale_weird < 13.8:
            return -np.inf
    else:
        Teff_err, logg_err = params

    if Teff_err < 0 or logg_err < 0:
        return -np.inf

    log_priors = [
        #inverse gamma on variance with Jeffrey's prior
        -3*log(Teff_err) - 0.5/Teff_err**2 * 1.577E-4 #From WDJ1336AB
        -3*log(logg_err) - 0.5/logg_err**2 * 2.978E-3 #From WDJ1336AB
    ]

    return sum(log_priors) + logprior_IFMR(IFMR)

def setup_params_IFMR(all_params, ifmr_x, outliers=False):
    """
    Takes a full array of parameters, and removes those corresponding
    to IFMR y-values, instead returning a reduced set of parameters
    and an IFMR object.
    """
    params, ifmr_y = np.split(all_params, [4 if outliers else 2])
    return params, IFMR_cls(ifmr_x, ifmr_y)

def logpost_DWD(all_params, DWD, ifmr_x):
    """
    Test posterior for fitting a single DWD only
    """
    Mi12, IFMR = setup_params_IFMR(all_params, ifmr_x)
    covMdtau = DWD.covMdtau_systematics(0.01, 0.01)
    if not np.isfinite(lp := logprior_DWD(Mi12, IFMR)):
        return -np.inf
    if not np.isfinite(ll := loglike_Mi12(Mi12, DWD.vecMdtau, covMdtau, IFMR)):
        return -np.inf
    return lp + ll

def logpost_DWDs(all_params, DWDs, ifmr_x, outliers=False):
    """
    Posterior distribution for fitting IFMR to all DWDs
    """
    params, IFMR = setup_params_IFMR(all_params, ifmr_x, outliers)
    if not np.isfinite(lp := logprior_DWDs(params, IFMR, outliers)):
        return -np.inf
    if not np.isfinite(ll := loglike_DWDs(params, DWDs, IFMR, outliers)):
        return -np.inf
    return lp + ll
