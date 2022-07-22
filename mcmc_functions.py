"""
MCMC routines for IFMR fitting with DWD binaries
"""
from math import log
import numpy as np
import numba
from scipy import stats
from scipy.special import logsumexp
from IFMR_tools import IFMR_cls, MSLT

MONOTONIC_IFMR = True
MONOTONIC_MASS_LOSS = True
DIRECT_MI_INTEGRATION = False
N_MARGINALISE = 1600
OUTLIER_DTAU_DIST = "normal" #one of ['normal', 'logit normal', 'uniform', 'beta']

if not MONOTONIC_IFMR and not DIRECT_MI_INTEGRATION:
    raise ValueError("Cannot fit non-monotonic IFMR with integration over Mf")

##################################
# constants
log_weights_uniform = 2*log(8-0.6)

def get_outlier_dtau_distribution(dist_name):
    """
    Choice of distributions for dtau when considering outliers. Valid choices
    are:
        - "normal"
        - "logit-normal"
        - "uniform"
        - "beta"
    In all cases, arguments are the dtau array, and the distibution scale (this
    has no effect for the uniform distribution, other than simplying the prior).
    """
    def outlier_norm(dtau, loc, scale):
        return stats.norm.logpdf(dtau, loc=loc, scale=scale)

    def outlier_logit_norm(dtau, loc, scale):
        z = 0.5*(dtau/13.8 + 1)
        logit_z = np.log(z/(1-z))
        z_mu = 0.5*(loc/13.8 + 1)
        logit_z_mu = np.log(z_mu/(1-z_mu))
        return stats.norm.logpdf(logit_z, loc=logit_z_mu, scale=scale) \
            - np.log(2*13.8*z*(1-z))

    def outlier_uniform(dtau, loc, scale):
        return stats.uniform.logpdf(dtau, loc=loc-scale, scale=2*scale)

    def outlier_beta(dtau, loc, scale):
        return stats.beta.logpdf(dtau, scale, scale, loc=loc-13.8, scale=2*13.8)

    outlier_dist_options = {
        'normal' : outlier_norm,
        'logit normal' : outlier_logit_norm,
        'uniform' : outlier_uniform,
        'beta' : outlier_beta,
    }
    return outlier_dist_options[dist_name]

def loglike_Mi12(Mi12, vec, cov, IFMR, outliers=False, scale_weird=None):
    """
    Computes the likelihood of an IFMR and initial masses for parameters
    of final masses and difference in WD cooling ages and their covariance.
    This is optionally computed for either the coeval or outlier distributions.
    """
    tau1_ms, tau2_ms = MSLT(Mi12)
    dtau_cool = tau2_ms-tau1_ms
    Mf1, Mf2 = IFMR(Mi12)
    X = np.array([Mf1, Mf2, dtau_cool])
    if outliers:
        ll_Mf12 = stats.multivariate_normal.logpdf(X[:2].T, mean=vec[:2], cov=cov[:2,:2])
        ll_dtau = outlier_dtau_dist(dtau_cool, vec[2], scale_weird)
        return ll_Mf12 + ll_dtau
    return stats.multivariate_normal.logpdf(X.T, mean=vec, cov=cov)

def loglike_Mi12_mixture(Mi12, vec, cov, IFMR, P_weird, scale_weird, separate=False):
    """
    Computes the likelihood of an IFMR and initial masses for parameters
    of final masses and difference in WD cooling ages and their covariance,
    and under the assumption of a fraction of systems being outliers drawn
    from an alternative distribution.
    """
    args = Mi12, vec, cov, IFMR
    logL_coeval  = loglike_Mi12(*args, outliers=False) + log(1-P_weird)
    logL_weird = loglike_Mi12(*args, outliers=True, scale_weird=scale_weird) + log(P_weird)

    if isinstance(logL_coeval, np.ndarray):
        nan_coeval, nan_weird = np.isnan(logL_coeval), np.isnan(logL_weird)
        logL_coeval[nan_coeval] = -np.inf
        logL_weird[nan_weird] = -np.inf
    else:
        if np.isnan(logL_coeval):
            logL_coeval = -np.inf
        if np.isnan(logL_weird):
            logL_weird = -np.inf

    if separate:
        return logL_coeval, logL_weird
    return np.logaddexp(logL_coeval, logL_weird)

@numba.vectorize
def logprior_Mi12(Mi1, Mi2):
    """
    Priors on inital masses
    """
    if not (0.6 < Mi1 < 8.0 and 0.6 < Mi2 < 8.0):
        return -np.inf
    return -2.3*log(Mi1*Mi2) #Kroupa IMF for m>0.5Msun

def loglike_DWD(params, DWD, IFMR, outliers=False):
    """
    Marginal distribution :
    P(DWD | theta) = \\iint P(Mi1, Mi2, DWD | theta) dMi1 dMi2
    """
    if outliers:
        P_weird, scale_weird, Teff_err, logg_err = params
    else:
        Teff_err, logg_err = params
    covMdtau = DWD.covMdtau_systematics(Teff_err, logg_err)
    vecM, covM = DWD.vecMdtau[:2], covMdtau[:2,:2]

    if DIRECT_MI_INTEGRATION:
        Mi12 = np.random.uniform(0.6, 8, (2, N_MARGINALISE))
    else:
        Mi12, Mf12 = IFMR.draw_Mi_samples(vecM, covM, N_MARGINALISE)
        if len(Mf12) <= 1:
            return -np.inf
        log_weights = -stats.multivariate_normal.logpdf(Mf12, mean=vecM, cov=covM)
        jac1, jac2 = IFMR.inv_grad(Mf12).T

    #importance sampling
    if outliers:
        log_like = loglike_Mi12_mixture(Mi12, DWD.vecMdtau, covMdtau, \
            IFMR, P_weird, scale_weird)
    else:
        log_like = loglike_Mi12(Mi12, DWD.vecMdtau, covMdtau, IFMR)
    log_probs = logprior_Mi12(*Mi12) + log_like

    if DIRECT_MI_INTEGRATION:
        return logsumexp(log_probs+log_weights_uniform) - log(N_MARGINALISE)
    return logsumexp(log_probs+log_weights, b=np.abs(jac1*jac2)) - log(N_MARGINALISE)

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
        P_weird, scale_weird, Teff_err, logg_err = params
        if not 0 < P_weird < 1:
            return -np.inf
        if scale_weird < 0:
            return -np.inf
    else:
        Teff_err, logg_err = params

    if Teff_err < 0:
        return -np.inf
    if logg_err < 0:
        return -np.inf

    #Mass loss, q must be 0 < q < 1
    if not all(0 < q < 1 for q in IFMR.Mf_Mi):
        return -np.inf

    #piecewise points in IFMR must be increasing
    if MONOTONIC_IFMR and (not np.allclose(IFMR.y, np.sort(IFMR.y)) \
    or MONOTONIC_MASS_LOSS and not \
    np.allclose(IFMR.mass_loss, np.sort(IFMR.mass_loss))):
        return -np.inf

    log_priors = [
        #stats.arcsine.logpdf(IFMR.Mf_Mi).sum(),
        #-log(Teff_err),
        #-log(logg_err),
        -0.5*((Teff_err-0.03)/0.001)**2,
        -0.5*((logg_err-0.03)/0.001)**2,
    ]

    if outliers:
        log_priors += [
            stats.arcsine.logpdf(P_weird),
            #stats.rayleigh.logpdf(scale_weird, scale=1),
            -log(scale_weird),
        ]

    return sum(log_priors)

def setup_params_IFMR(all_params, ifmr_x, outliers=False):
    """
    Takes a full set of parameters, and removes those corresponding
    to IFMR y-values, instead returning a reduced set of parameters
    and an IFMR object.
    """
    if outliers:
        P_weird, scale_weird, Teff_err, logg_err, *ifmr_y = all_params
        params = P_weird, scale_weird, Teff_err, logg_err
    else:
        Teff_err, logg_err, *ifmr_y = all_params
        params = Teff_err, logg_err
    return params, IFMR_cls(ifmr_x, ifmr_y)

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

outlier_dtau_dist = get_outlier_dtau_distribution(OUTLIER_DTAU_DIST)
