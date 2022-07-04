import numpy as np
import numba
from scipy.interpolate import interp1d
from scipy import stats
from math import log
from misc import pairwise

MONOTONIC_IFMR = False
N_MARGINALISE = 1600
OUTLIER_DTAU_DIST = "normal" #one of ['normal', 'logit normal', 'uniform', 'beta']

#MS lifetime from MESA data
M_init, t_pre = np.loadtxt("MESA_lifetime.dat", unpack=True, skiprows=1)
t_pre /= 1e9 #to Gyr
log_tau_fun = interp1d(M_init, np.log10(t_pre), kind='cubic', bounds_error=False)

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
        z = 0.5*(dtau/13.8+1)
        logit_z = np.log(z/(1-z))
        return stats.norm.logpdf(logit_z, loc=0, scale=scale) - np.log(2*13.8*(z*(1-z)))

    def outlier_uniform(dtau, scale):
        return stats.uniform.logpdf(dtau, -13.8, 13.8)

    def outlier_beta(dtau, scale):
        return stats.beta.logpdf(dtau, scale_weird, scale, loc=-13.8, scale=2*13.8)

    outlier_dist_options = {
        'normal' : outlier_norm,
        'logit normal' : outlier_logit_norm,
        'uniform' : outlier_uniform,
        'beta' : outlier_beta,
    }
    return outlier_dist_options[dist_name]

outlier_dtau_dist = get_outlier_dtau_distribution(OUTLIER_DTAU_DIST)

def MSLT(Mi):
    """
    Main sequence lifetime
    """
    return 10**log_tau_fun(Mi)

def loglike_Mi12(Mi1, Mi2, vec, cov, IFMR, outliers=False, scale_weird=None):
    """
    params are the MS star masses and the y values of the IFMR at values. DWD
    contains a vector of Mf1, Mf2, dtau_cool and the corresponding covariance
    matrix.
    """
    tau1, tau2 = MSLT([Mi1, Mi2])
    dtau = tau2-tau1
    Mf1, Mf2 = IFMR([Mi1, Mi2])
    X = np.array([Mf1, Mf2, dtau])
    if outliers:
        ll1 =  stats.multivariate_normal.logpdf(X[:2].T, mean=vec[:2], cov=cov[:2,:2])
        ll2 = outlier_dtau_dist(dtau, scale_weird)
        return ll1 + ll2
    else:
        return stats.multivariate_normal.logpdf(X.T, mean=vec, cov=cov)

def loglike_Mi12_outliers(Mi1, Mi2, vec, cov, IFMR, P_weird, scale_weird, separate=False):
    """
    P_weird is the probability that any DWD is weird, scale_weird is the
    variance of cooling age differences for weird DWDs. The likelihood
    is a mixture model of the coeval and non-coeval likelihoods.
    """
    args = Mi1, Mi2, vec, cov, IFMR
    logL_coeval  = loglike_Mi12(*args, outliers=False) + log(1-P_weird)
    logL_weird = loglike_Mi12(*args, outliers=True, scale_weird=scale_weird) + log(P_weird)

    if separate:
        return logL_coeval, logL_weird
    else:
        return np.logaddexp(logL_coeval, logL_weird)
        

@numba.vectorize
def logprior_Mi12(Mi1, Mi2):
    """
    priors on inital masses
    """
    if not (0.6 < Mi1 < 8) or not (0.6 < Mi2 < 8):
        return -np.inf
    return -2.35*log(Mi1*Mi2) #Lazy Salpeter IMF as initial mass prior

def logpost_Mi12(Mi1, Mi2, vec, cov, IFMR):
    lp = logprior_Mi12(Mi1, Mi2)
    ll = loglike_Mi12(Mi1, Mi2, vec, cov, IFMR)
    return lp + ll

def logpost_Mi12_outliers(Mi1, Mi2, vec, cov, IFMR, P_weird=None, scale_weird=None):
    lp = logprior_Mi12(Mi1, Mi2)
    ll = loglike_Mi12_outliers(Mi1, Mi2, vec, cov, IFMR, P_weird, scale_weird)
    return lp + ll

def grad_IFMR_i(Mf, IFMR):
    """
    The gradient of the inverse IFMR, for a piecewise
    linear and piecewise continuous IFMR. This essentially
    gets the jacobian dMi/dMf
    """
    segments = [Mf < y for y in IFMR.y]
    grads = [(x1-x0)/(y1-y0) for (x0, x1), (y0, y1)
        in zip(pairwise(IFMR.x), pairwise(IFMR.y))]
    grads.insert(0, 0)
    return np.select(segments, grads)

def loglike_DWD(params, DWD, IFMR, IFMR_i, outliers=False):
    """
    Marginal distribution :
    P(theta, DWD) = \iint P(Mi1, Mi2, theta | DWD) dMi1 dMi2
    """
    if outliers:
        P_weird, scale_weird, Teff_err, logg_err = params
    else:
        Teff_err, logg_err = params
    covMdtau = DWD.covMdtau_systematics(Teff_err, logg_err)
    vecM, covM = DWD.vecMtau[:2], covMdtau[:2,:2]

    #draw samples of Mf12, and calc Mi12, and jacobians
    Mf12 = np.random.multivariate_normal(vecM, covM, N_MARGINALISE)
    ok = (Mf12 > IFMR.y.min()) & (Mf12 < IFMR.y.max()) #reject samples outside of IFMR
    ok = np.all(ok, axis=1)
    Mf12 = Mf12[ok,:]
    Mi12 = IFMR_i(Mf12)
    ok = (Mi12 > 0.6) & (Mi12 < 8.0) #MSLT table limits
    ok = np.all(ok, axis=1)
    (Mi1, Mi2), Mf12 = Mi12[ok,:].T, Mf12[ok,:]
    jac1, jac2 = grad_IFMR_i(Mf12, IFMR).T

    #importance sampling
    if outliers:
        log_probs = logpost_Mi12_outliers(Mi1, Mi2, DWD.vecMdtau, covMdtau, \
            IFMR, P_weird, scale_weird)
    else:
        log_probs = logpost_Mi12(Mi1, Mi2, DWD.vecMdtau, covMdtau, IFMR)
    log_weights = -stats.multivariate_normal.logpdf(Mf12, mean=vecM, cov=covM)
    integrand = np.exp(log_probs + log_weights) * jac1 * jac2
    I = np.mean(integrand)

    return log(I) if I > 0 and np.isfinite(I) else -np.inf 

def loglike_DWDs(theta, DWDs, ifmr_x, outliers=False):
    """
    log likelihood for ifmr_y for all DWDs
    """
    if outliers:
        P_weird, scale_weird, Teff_err, logg_err, *ifmr_y = theta
        params = P_weird, scale_weird, Teff_err, logg_err
    else:
        Teff_err, logg_err, *ifmr_y = theta
        params = Teff_err, logg_err
    IFMR, IFMR_i = interp1d(ifmr_x, ifmr_y), interp1d(ifmr_y, ifmr_x)
    return sum(loglike_DWD(params, DWD, IFMR, IFMR_i, outliers=outliers) \
            for DWD in DWDs)

def logprior(params, ifmr_x, outliers=False):
    """
    priors on IFMR all ifmr parameters
    """
    if outliers:
        P_weird, scale_weird, Teff_err, logg_err, *ifmr_y = params
        if not (0 < P_weird < 1):
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
    mf_mi = [y/x for x, y in zip(ifmr_x, ifmr_y)]
    if not all(0 < q < 1 for q in mf_mi):
        return -np.inf

    #piecewise points in IFMR must be increasing
    if MONOTONIC_IFMR and list(ifmr_y) != sorted(ifmr_y):
        return -np.inf

    log_priors = [
        stats.arcsine.logpdf(mf_mi).sum(),
        -log(Teff_err),
        -log(logg_err),
        stats.arcsine.logpdf(P_weird) if outliers else 0,
        #stats.rayleigh.logpdf(scale_weird, scale=1) if outliers else 0,
        -log(scale_weird),
    ]

    return sum(log_priors)

def logpost_DWD(params, DWD, ifmr_x):
    """
    Test posterior for fitting a single DWD only
    """
    lp = logprior(params, ifmr_x)
    return lp if lp == -np.inf else lp + loglike_DWD(params, DWD, ifmr_x)

def logpost_DWDs(params, DWDs, ifmr_x, outliers=False):
    """
    Posterior distribution for fitting IFMR to all DWDs
    """
    lp = logprior(params, ifmr_x, outliers)
    return lp if lp == -np.inf else lp + loglike_DWDs(params, DWDs, ifmr_x, outliers)
