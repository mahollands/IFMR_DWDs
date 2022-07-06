"""
MCMC routines for IFMR fitting with DWD binaries
"""
import numpy as np
import numba
from scipy import stats
from math import log
from misc import create_IFMR, draw_Mi_samples, MSLT

MONOTONIC_IFMR = True
N_MARGINALISE = 1600
OUTLIER_DTAU_DIST = "normal" #one of ['normal', 'logit normal', 'uniform', 'beta']

def get_outlier_dtau_distribution(dist_name):
    """
    Choice of distributions for dtau when considering outliers. Valid choices
    are:
        - "normal"
        - "logit-normal"
        - "uniform"
        - "beta"
    In all cases, arguments are the dtau array, and the distibution scale (this
    has no effect for the uniform distribution.
    """
    def outlier_norm(dtau, scale):
        return stats.norm.logpdf(dtau, loc=0, scale=scale)

    def outlier_logit_norm(dtau, scale):
        z = 0.5*(dtau/13.8 + 1)
        logit_z = np.log(z/(1-z))
        return stats.norm.logpdf(logit_z, loc=0, scale=scale) - np.log(2*13.8*(z*(1-z)))

    def outlier_uniform(dtau, scale):
        return stats.uniform.logpdf(dtau, -13.8, 2*13.8)

    def outlier_beta(dtau, scale):
        return stats.beta.logpdf(dtau, scale, scale, loc=-13.8, scale=2*13.8)

    outlier_dist_options = {
        'normal' : outlier_norm,
        'logit normal' : outlier_logit_norm,
        'uniform' : outlier_uniform,
        'beta' : outlier_beta,
    }
    return outlier_dist_options[dist_name]

outlier_dtau_dist = get_outlier_dtau_distribution(OUTLIER_DTAU_DIST)

def loglike_Mi12(Mi12, vec, cov, IFMR, outliers=False, scale_weird=None):
    """
    params are the MS star masses and the y values of the IFMR at values. DWD
    contains a vector of Mf1, Mf2, dtau_cool and the corresponding covariance
    matrix.
    """
    tau1, tau2 = MSLT(Mi12)
    dtau = tau2-tau1
    Mf1, Mf2 = IFMR(Mi12)
    X = np.array([Mf1, Mf2, dtau])
    if outliers:
        ll1 =  stats.multivariate_normal.logpdf(X[:2].T, mean=vec[:2], cov=cov[:2,:2])
        ll2 = outlier_dtau_dist(dtau, scale_weird)
        return ll1 + ll2
    return stats.multivariate_normal.logpdf(X.T, mean=vec, cov=cov)

def loglike_Mi12_outliers(Mi12, vec, cov, IFMR, P_weird, scale_weird, separate=False):
    """
    P_weird is the probability that any DWD is weird, scale_weird is the
    variance of cooling age differences for weird DWDs. The likelihood
    is a mixture model of the coeval and non-coeval likelihoods.
    """
    args = Mi12, vec, cov, IFMR
    logL_coeval  = loglike_Mi12(*args, outliers=False) + log(1-P_weird)
    logL_weird = loglike_Mi12(*args, outliers=True, scale_weird=scale_weird) + log(P_weird)

    if separate:
        return logL_coeval, logL_weird
    return np.logaddexp(logL_coeval, logL_weird)

@numba.vectorize
def logprior_Mi12(Mi1, Mi2):
    """
    priors on inital masses
    """
    return -2.35*log(Mi1*Mi2) #Lazy Salpeter IMF as initial mass prior

def logpost_Mi12(Mi12, vec, cov, IFMR):
    """
    Posterior probablity without outliers for a DWD with Mi1 and Mi2 samples.
    """
    lp = logprior_Mi12(*Mi12)
    ll = loglike_Mi12(Mi12, vec, cov, IFMR)
    return lp + ll

def logpost_Mi12_outliers(Mi12, vec, cov, IFMR, P_weird=None, scale_weird=None):
    """
    Posterior probablity with outliers for a DWD with Mi1 and Mi2 samples.
    """
    lp = logprior_Mi12(*Mi12)
    ll = loglike_Mi12_outliers(Mi12, vec, cov, IFMR, P_weird, scale_weird)
    return lp + ll

def loglike_DWD(params, DWD, IFMR, outliers=False):
    """
    Marginal distribution :
    P(theta | DWD) = \\iint P(Mi1, Mi2, theta | DWD) dMi1 dMi2
    """
    if outliers:
        P_weird, scale_weird, Teff_err, logg_err = params
    else:
        Teff_err, logg_err = params
    covMdtau = DWD.covMdtau_systematics(Teff_err, logg_err)
    vecM, covM = DWD.vecMdtau[:2], covMdtau[:2,:2]

    Mi12, Mf12 = draw_Mi_samples(vecM, covM, IFMR, N_MARGINALISE)
    if len(Mf12) == 0:
        return -np.inf
    jac1, jac2 = IFMR.inv_grad(Mf12).T

    #importance sampling
    if outliers:
        log_probs = logpost_Mi12_outliers(Mi12, DWD.vecMdtau, covMdtau, \
            IFMR, P_weird, scale_weird)
    else:
        log_probs = logpost_Mi12(Mi12, DWD.vecMdtau, covMdtau, IFMR)
    log_weights = -stats.multivariate_normal.logpdf(Mf12, mean=vecM, cov=covM)
    integrand = np.exp(log_probs + log_weights) * jac1 * jac2
    I = np.mean(integrand)

    return log(I) if I > 0 and np.isfinite(I) else -np.inf

def loglike_DWDs(params, DWDs, IFMR, outliers=False):
    """
    log likelihood for ifmr_y for all DWDs
    """
    return sum(loglike_DWD(params, DWD, IFMR, outliers=outliers) for DWD in DWDs)

def logprior(params, IFMR, outliers=False):
    """
    priors on IFMR all ifmr parameters
    """
    if outliers:
        P_weird, scale_weird, Teff_err, logg_err, *ifmr_y = params
        if not 0 < P_weird < 1:
            return -np.inf
        if scale_weird < 0:
            return -np.inf
    else:
        Teff_err, logg_err, *ifmr_y = params

    if Teff_err < 0:
        return -np.inf
    if logg_err < 0:
        return -np.inf

    #Mass loss, q must be 0 < q < 1
    if not all(0 < q < 1 for q in IFMR.mf_mi):
        return -np.inf

    #piecewise points in IFMR must be increasing
    if MONOTONIC_IFMR and not np.allclose(IFMR.y, np.sort(IFMR.y)):
        return -np.inf

    log_priors = [
        stats.arcsine.logpdf(IFMR.mf_mi).sum(),
        -log(Teff_err),
        -log(logg_err),
    ]

    if outliers:
        log_priors += [
            stats.arcsine.logpdf(P_weird) if outliers else 0,
            stats.rayleigh.logpdf(scale_weird, scale=1) if outliers else 0,
            #-log(scale_weird),
        ]

    return sum(log_priors)

def setup_params_IFMR(all_params, ifmr_x, outliers=False):
    if outliers:
        P_weird, scale_weird, Teff_err, logg_err, *ifmr_y = all_params
        params = P_weird, scale_weird, Teff_err, logg_err
    else:
        Teff_err, logg_err, *ifmr_y = all_params
        params = Teff_err, logg_err
    return params, create_IFMR(ifmr_x, ifmr_y)

def logpost_DWD(all_params, DWD, ifmr_x):
    """
    Test posterior for fitting a single DWD only
    """
    params, IFMR = setup_params_IFMR(all_params, ifmr_x)
    lp = logprior(params, IFMR)
    return lp if lp == -np.inf else lp + loglike_DWD(params, DWD, IFMR)

def logpost_DWDs(all_params, DWDs, ifmr_x, outliers=False):
    """
    Posterior distribution for fitting IFMR to all DWDs
    """
    params, IFMR = setup_params_IFMR(all_params, ifmr_x, outliers)
    lp = logprior(params, IFMR, outliers)
    return lp if lp == -np.inf else lp + loglike_DWDs(params, DWDs, IFMR, outliers)
