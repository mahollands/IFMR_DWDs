import numpy as np
import numba
from scipy.interpolate import interp1d
from scipy import stats
from math import log
from misc import pairwise

MONOTONIC_IFMR = False
N_MARGINALISE = 400

#MS lifetime from MESA data
M_init, t_pre = np.loadtxt("MESA_lifetime.dat", unpack=True, skiprows=1)
t_pre /= 1e9 #to Gyr
log_tau_fun = interp1d(M_init, np.log10(t_pre), kind='cubic', bounds_error=False)

def MSLT(Mi):
    """
    Main sequence lifetime
    """
    return 10**log_tau_fun(Mi)

def get_Mf12_dtau(DWD, Teff_err, logg_err):
    """
    Add systematic uncertainties to Teff and logg covariance matrix, then get
    M1f, M2f and dtau vector and covariance matrix via linear transform.
    """
    Tg_vec, Tg_cov, vec, Jac = DWD
    T10, T20, g10, g20 = Tg_vec
    err_syst = np.array([Teff_err*T10, Teff_err*T20, logg_err, logg_err])
    Tg_cov_ = Tg_cov + np.diag(err_syst**2)
    cov = Jac @ Tg_cov_ @ Jac.T
    return vec, cov

def loglike_Mi12(Mi1, Mi2, vec, cov, IFMR, outliers=False, V_weird=None):
    """
    params are the MS star masses and the y values of the IFMR at values. DWD
    contains a vector of Mf1, Mf2, dtau_cool and the corresponding covariance
    matrix.
    """
    if outliers:
        cov_ = np.zeros((3, 3))
        cov_[:2, :2] = cov[:2, :2]
        cov_[2, 2] = V_weird
    else:
        cov_ = cov
    Mf1, Mf2 = IFMR([Mi1, Mi2])
    tau1, tau2 = MSLT([Mi1, Mi2])
    X = np.array([Mf1, Mf2, tau2-tau1])
    return stats.multivariate_normal.logpdf(X.T, mean=vec, cov=cov_)

def loglike_Mi12_outliers(Mi1, Mi2, vec, cov, IFMR, P_weird, V_weird, separate=False):
    """
    P_weird is the probability that any DWD is weird, V_weird is the
    variance of cooling age differences for weird DWDs. The likelihood
    is a mixture model of the coeval and non-coeval likelihoods.
    """
    args = Mi1, Mi2, vec, cov, IFMR
    logL_coeval  = loglike_Mi12(*args, outliers=False) + log(1-P_weird)
    logL_weird = loglike_Mi12(*args, outliers=True, V_weird=V_weird) + log(P_weird)

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

def logpost_Mi12_outliers(Mi1, Mi2, vec, cov, IFMR, P_weird=None, V_weird=None):
    lp = logprior_Mi12(Mi1, Mi2)
    ll = loglike_Mi12_outliers(Mi1, Mi2, vec, cov, IFMR, P_weird, V_weird)
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
        P_weird, V_weird, Teff_err, logg_err = params
    else:
        Teff_err, logg_err = params
    vecMtau, covMtau = get_Mf12_dtau(DWD, Teff_err, logg_err)
    vecM, covM = vecMtau[:2], covMtau[:2,:2]

    #draw samples of Mf12, and calc Mi12, and jacobians
    Mf12 = np.random.multivariate_normal(vecM, covM, N_MARGINALISE)
    ok = (Mf12 > IFMR.y.min()) & (Mf12 < IFMR.y.max()) #reject samples outside of IFMR
    ok = np.all(ok, axis=1)
    Mf12 = Mf12[ok,:]
    Mi1, Mi2 = IFMR_i(Mf12.T)
    jac1, jac2 = grad_IFMR_i(Mf12, IFMR).T

    #importance sampling
    if outliers:
        log_probs = logpost_Mi12_outliers(Mi1, Mi2, vecMtau, covMtau, IFMR, \
            P_weird, V_weird)
    else:
        log_probs = logpost_Mi12(Mi1, Mi2, vecMtau, covMtau, IFMR)
    log_weights = -stats.multivariate_normal.logpdf(Mf12, mean=vecM, cov=covM)
    integrand = np.exp(log_probs + log_weights) * jac1 * jac2
    I = np.mean(integrand)

    return log(I) if I > 0 and np.isfinite(I) else -np.inf 

def loglike_DWDs(theta, DWDs, ifmr_x, outliers=False):
    """
    log likelihood for ifmr_y for all DWDs
    """
    if outliers:
        P_weird, V_weird, Teff_err, logg_err, *ifmr_y = theta
        params = P_weird, V_weird, Teff_err, logg_err
    else:
        Teff_err, logg_err, *ifmr_y = theta
        params = Teff_err, logg_err
    IFMR, IFMR_i = interp1d(ifmr_x, ifmr_y), interp1d(ifmr_y, ifmr_x)
    return sum(loglike_DWD(params, DWD, IFMR, IFMR_i, outliers=outliers) \
            for DWD in DWDs.values())

def logprior(params, ifmr_x, outliers=False):
    """
    priors on IFMR all ifmr parameters
    """
    if outliers:
        P_weird, V_weird, Teff_err, logg_err, *ifmr_y = params
        if not (0 < P_weird < 1):
            return -np.inf
        if V_weird < 0:
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
        - log(Teff_err),
        - log(logg_err),
        stats.arcsine.logpdf(P_weird) if outliers else 0,
        - log(V_weird) if outliers else 0,
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
