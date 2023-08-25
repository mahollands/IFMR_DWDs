"""
MCMC routines for IFMR fitting with DWD binaries
"""
from math import log, log1p
from functools import partial
import numpy as np
import numba
from scipy import stats
from scipy.special import logsumexp, erfc
from IFMR_tools import IFMR_cls, MSLT
from misc import is_sorted
import IFMR_config as conf 

##################################
# constants
log_weights_uniform = 2*log(8-0.5)
t_universe = 13.8

def loglike_Mf12(Mf12, tau12_preWD, Mdtau, covMdtau, IFMR, scale_outlier=None):
    """
    Computes the likelihood of an IFMR and initial masses for one DWD with
    measured final masses and difference in WD cooling ages (and their
    covariance). This is optionally computed for either the coeval or outlier
    distributions.
    """
    tau1_preWD, tau2_preWD = tau12_preWD
    X, cov_ = np.vstack([Mf12, tau2_preWD-tau1_preWD]), np.copy(covMdtau)
    if scale_outlier is not None:
        cov_[2,2] += scale_outlier**2
    return stats.multivariate_normal.logpdf(X.T, mean=Mdtau, cov=cov_)

def loglike_Mf12_mixture(Mf12, tau12_preWD, Mdtau, covMdtau, IFMR, P_outlier, \
    scale_outlier, return_logL0=False):
    """
    Computes the likelihood of an IFMR and initial masses for one DWD with
    measured final masses and difference in WD cooling ages (and their
    covariance), and under the assumption of a fraction of systems being
    outliers drawn from a broader dtau_cool distribution.
    """
    args = Mf12, tau12_preWD, Mdtau, covMdtau, IFMR
    logL0  = loglike_Mf12(*args)
    logL_outlier = loglike_Mf12(*args, scale_outlier=scale_outlier)

    if isinstance(logL0, np.ndarray):
        logL0[np.isnan(logL0)] = -np.inf
        logL_outlier[np.isnan(logL_outlier)] = -np.inf
    else:
        if np.isnan(logL0):
            logL0 = -np.inf
        if np.isnan(logL_outlier):
            logL_outlier = -np.inf

    logL0 += log1p(-P_outlier)
    logL_outlier += log(P_outlier)
    logL_total = np.logaddexp(logL0, logL_outlier)

    if return_logL0:
        return logL_total, logL0
    return logL_total

@numba.vectorize
def logprior_Mi12(Mi1, Mi2):
    """
    Priors on inital masses, i.e.
    P(Mi1, Mi2) = (M1*M2)**-2.3
    """
    if 0.5 < Mi1 < 8.0 and 0.5 < Mi2 < 8.0:
        return -2.3*log(Mi1*Mi2) #Kroupa IMF for m>0.5Msun
    return -np.inf

def logprior_tau12(tau12_preWD, tau12_cool, tau12_cov):
    """
    Prior on total lifetime. Doesn't take into account covariance for
    speed reasons.
    """
    tau12_total = tau12_preWD + tau12_cool[:,np.newaxis]
    tau12_var = np.diag(tau12_cov)[:,np.newaxis]
    p1, p2 = erfc((tau12_total-t_universe)/np.sqrt(2*tau12_var))
    return np.log(np.abs(p1*p2))

def loglike_DWD(hyper_params, DWD, IFMR, outliers=False, return_logL0=False):
    """
    log of marginal distribution:
    P(DWD | IFMR, theta) = \\iint P(Mi1, Mi2, DWD | IFMR, theta) dMi1 dMi2
    """
    if outliers:
        P_outlier, scale_outlier, Teff_err, logg_err = hyper_params
        loglike_Mf12_ = partial(loglike_Mf12_mixture, \
            P_outlier=P_outlier, scale_outlier=scale_outlier)
    else:
        Teff_err, logg_err = hyper_params
        loglike_Mf12_ = loglike_Mf12
    covMdtau = DWD.covMdtau_systematics(Teff_err, logg_err)

    if conf.MONOTONIC_IFMR:
        covM = covMdtau[:2,:2]
        Mf12 = DWD.draw_Mf_samples(covM, IFMR, conf.N_MARGINALISE)
        if Mf12.shape[1] <= 1:
            return -np.inf
        log_weights = -stats.multivariate_normal.logpdf(Mf12.T, mean=DWD.M12, cov=covM)
        Mi12 = IFMR.inv(Mf12)
        jac1, jac2 = IFMR.inv_grad(Mf12)
    else:
        Mi12 = np.random.uniform(0.5, 8, (2, conf.N_MARGINALISE))
        log_weights = log_weights_uniform
        Mf12 = IFMR(Mi12)
        jac1, jac2 = 1, 1

    #priors
    tau12_preWD = MSLT(Mi12)
    log_prior = np.zeros(Mi12.shape[1])
    if conf.MI_PRIOR:
        log_prior += logprior_Mi12(*Mi12)
    if conf.TAU_PRIOR:
        covtau = DWD.covMtau_systematics(Teff_err, logg_err)[2:,2:]
        log_prior += logprior_tau12(tau12_preWD, DWD.tau12, covtau)

    #importance sampling
    if outliers and return_logL0:
        log_like = loglike_Mf12_(Mf12, tau12_preWD, DWD.Mdtau, covMdtau, IFMR, \
            return_logL0=True)
        log_integrand = log_prior[np.newaxis,:] + np.array(log_like) \
        + log_weights[np.newaxis,:]
        return logsumexp(log_integrand, axis=1, b=np.abs(jac1*jac2)) \
            - log(conf.N_MARGINALISE)

    log_like = loglike_Mf12_(Mf12, tau12_preWD, DWD.Mdtau, covMdtau, IFMR)
    log_integrand = log_prior + log_like + log_weights
    return logsumexp(log_integrand, b=np.abs(jac1*jac2)) - log(conf.N_MARGINALISE)

def loglike_DWDs(hyper_params, DWDs, IFMR, outliers=False):
    """
    log likelihood for IFMR (and hyperparams) for all DWDs
    """
    return sum(loglike_DWD(hyper_params, DWD, IFMR, outliers=outliers) for DWD in DWDs)

def logprior_IFMR(IFMR):
    """
    Prior on the IFMR only. Specifically this amounts to cuts on the 
    """
    if conf.STRICT_MASS_LOSS and not all(0 < q < 1 for q in IFMR.mass_loss):
        return -np.inf

    if conf.MCH_PRIOR and np.any(IFMR.y > 1.4):
        return -np.inf

    if conf.MONOTONIC_IFMR and not is_sorted(IFMR.y):
        return -np.inf

    if conf.MONOTONIC_MASS_LOSS and not is_sorted(IFMR.mass_loss):
        return -np.inf

    return 0

def logprior_DWD(Mi12, IFMR):
    """
    prior for only one DWD
    """
    return logprior_Mi12(*Mi12) + logprior_IFMR(IFMR)

def logprior_DWDs(hyper_params, IFMR, outliers=False):
    """
    priors on IFMR all and ifmr related parameters
    """
    if outliers:
        P_outlier, scale_outlier, Teff_err, logg_err = hyper_params
        if not 0 < P_outlier < 1 or not 0 < scale_outlier < t_universe:
            return -np.inf
    else:
        Teff_err, logg_err = hyper_params

    if Teff_err < 0 or logg_err < 0:
        return -np.inf

    log_priors = [
        #inverse gamma on variance with Jeffrey's prior
        -3*log(Teff_err) - 0.5/Teff_err**2 * conf.S_T
        -3*log(logg_err) - 0.5/logg_err**2 * conf.S_g
    ]

    return sum(log_priors) + logprior_IFMR(IFMR)

def setup_params_IFMR(all_params, ifmr_x, outliers=False):
    """
    Splits up a full array of parameters, returning a reduced set of parameters
    and an IFMR object.
    """
    hyper_params, ifmr_y = np.split(all_params, [4 if outliers else 2])
    return hyper_params, IFMR_cls(ifmr_x, ifmr_y)

def logpost_DWD(all_params, DWD, ifmr_x, Teff_err=0.01, logg_err=0.01):
    """
    Test posterior for fitting a single DWD only
    """
    Mi12, IFMR = setup_params_IFMR(all_params, ifmr_x)
    Mf12 = IFMR(Mi12)
    covMdtau = DWD.covMdtau_systematics(Teff_err, logg_err)
    if not np.isfinite(lp := logprior_DWD(Mi12, IFMR)):
        return -np.inf
    if not np.isfinite(ll := loglike_Mf12(Mf12, DWD.Mdtau, covMdtau, IFMR)):
        return -np.inf
    return lp + ll

def logpost_DWDs(all_params, DWDs, ifmr_x, outliers=False):
    """
    Posterior distribution for fitting an IFMR to a list of DWDs
    """
    hyper_params, IFMR = setup_params_IFMR(all_params, ifmr_x, outliers)
    if not np.isfinite(lp := logprior_DWDs(hyper_params, IFMR, outliers)):
        return -np.inf
    if not np.isfinite(ll := loglike_DWDs(hyper_params, DWDs, IFMR, outliers)):
        return -np.inf
    return lp + ll
