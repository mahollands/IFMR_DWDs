"""
Routines and classes for working with Initial-Final Mass-Relations.
"""
from itertools import tee
import numpy as np
from scipy.interpolate import interp1d

#MS lifetime from MESA data
M_init, t_pre = np.loadtxt("MESA_lifetime.dat", unpack=True, skiprows=1)
t_pre /= 1e9 #to Gyr
log_tau_fun = interp1d(M_init, np.log10(t_pre), kind='cubic', bounds_error=False)

def MSLT(Mi):
    """
    Main sequence lifetime (Gyr) for an initial mass (Msun).
    """
    return 10**log_tau_fun(Mi)

def pairwise(iterable):
    """
    Roughly recreates the python 3.10 itertools.pairwise.
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class IFMR_cls(interp1d):
    """
    IFMR class for calculating final masses from initial masses
    and vice-versa. All masses in units of Msun.
    """
    def __init__(self, ifmr_x, ifmr_y):
        super().__init__(ifmr_x, ifmr_y)
        self.inv = interp1d(ifmr_y, ifmr_x)
        self.i_grads = np.diff(ifmr_x)/np.diff(ifmr_y)
        self.y_x = ifmr_y/ifmr_x

    def inv_grad(self, Mf):
        """
        The gradient of the inverse IFMR, for a piecewise
        linear and piecewise continuous IFMR. This essentially
        gets the jacobian dMi/dMf
        """
        segments = [Mf < y for y in self.y[1:]]
        return np.select(segments, self.i_grads)

    @property
    def Mi(self):
        """
        Mi is alias for x array storing initial masses
        """
        return self.x

    @property
    def Mf(self):
        """
        Mf is alias for y array storing initial masses
        """
        return self.y

    @property
    def Mf_Mi(self):
        """
        Mf_Mi is alias for y_x array storing ratio of Mf/Mi
        """
        return self.y_x

    @property
    def mass_loss(self):
        """
        Fraction of mass lost as a function of Mi
        """
        return 1-self.Mf_Mi


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
    return Mi12.T, Mf12
