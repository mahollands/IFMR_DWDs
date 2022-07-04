import numpy as np
from scipy.interpolate import interp1d

def pairwise(collection):
    """
    Roughly recreates the python 3.10 itertools.pairwise. Will only work for
    collections, not iterators.
    """
    return zip(collection, collection[1:])

def generate_IFMR(ifmr_x, ifmr_y):
    """
    Create an interpolation object for the IFMR. The inverse IFMR is contained
    within the .inv attribute.
    """
    IFMR = interp1d(ifmr_x, ifmr_y)
    IFMR.inv = interp1d(ifmr_y, ifmr_x)
    return IFMR

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

def draw_mass_samples(vecM, covM, IFMR, N_MARGINALISE):
    """
    Using central values for final masses and the joint covariance matrix
    calculate initial masses and the IFMR jacobian using an IFMR
    """
    Mf12 = np.random.multivariate_normal(vecM, covM, N_MARGINALISE)
    ok = (Mf12 > IFMR.y.min()) & (Mf12 < IFMR.y.max()) #reject samples outside of IFMR
    ok = np.all(ok, axis=1)
    Mf12 = Mf12[ok,:]
    Mi12 = IFMR.inv(Mf12)
    ok = (Mi12 > 0.6) & (Mi12 < 8.0) #MSLT table limits
    ok = np.all(ok, axis=1)
    return Mi12[ok,:].T, Mf12[ok,:]

