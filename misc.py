import numpy as np
from scipy.interpolate import interp1d
from itertools import tee

#MS lifetime from MESA data
M_init, t_pre = np.loadtxt("MESA_lifetime.dat", unpack=True, skiprows=1)
t_pre /= 1e9 #to Gyr
log_tau_fun = interp1d(M_init, np.log10(t_pre), kind='cubic', bounds_error=False)

def MSLT(Mi):
    """
    Main sequence lifetime
    """
    return 10**log_tau_fun(Mi)

def pairwise(iterable):
    """
    Roughly recreates the python 3.10 itertools.pairwise.
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class create_IFMR(interp1d):
    def __init__(self, ifmr_x, ifmr_y):
        super().__init__(ifmr_x, ifmr_y)
        self.inv = interp1d(ifmr_y, ifmr_x)
        self.i_grads = np.diff(ifmr_x)/np.diff(ifmr_y)
        self.mf_mi = ifmr_y/ifmr_x

    def inv_grad(self, Mf):
        """
        The gradient of the inverse IFMR, for a piecewise
        linear and piecewise continuous IFMR. This essentially
        gets the jacobian dMi/dMf
        """
        segments = [Mf < y for y in self.y[1:]]
        return np.select(segments, self.i_grads)

def draw_Mi_samples(vecM, covM, IFMR, N_MARGINALISE):
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

