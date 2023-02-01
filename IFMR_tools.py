"""
Routines and classes for working with Initial-Final Mass-Relations.
"""
from itertools import tee
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#MS lifetime from MESA data
M_init, t_pre = np.loadtxt("MESA_lifetime.dat", unpack=True, skiprows=1)
t_pre /= 1e9 #to Gyr
log_tau_fun = interp1d(np.log(M_init), np.log(t_pre), kind='linear', bounds_error=False)

def MSLT(Mi):
    """
    Main sequence lifetime (Gyr) for an initial mass (Msun).
    """
    return np.exp(log_tau_fun(np.log(Mi)))

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
        self.inv = interp1d(self.y, self.x)
        self.i_grads = np.diff(self.x)/np.diff(self.y)
        self.y_x = self.y/self.x

    def __repr__(self):
        return "IFMR_cls({}, {})".format(list(self.x), list(self.y))

    def __len__(self):
        return len(self.x)

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

    def draw_Mi_samples(self, vecM, covM, N_MARGINALISE):
        """
        Using central values for final masses and the joint covariance matrix
        calculate initial masses and the IFMR jacobian using an IFMR
        """
        Mf12 = np.random.multivariate_normal(vecM, covM, N_MARGINALISE)
        ok = (Mf12 > self.y.min()) & (Mf12 < self.y.max()) #reject samples outside of IFMR
        ok = np.all(ok, axis=1)
        Mf12 = Mf12[ok,:]
        Mi12 = self.inv(Mf12)
        return Mi12.T, Mf12

    def plot(self, kind='y', *args, **kwargs):
        """
        Plot the IFMR
        """
        y_plot = getattr(self, kind)
        plt.plot(self.x, y_plot, *args, **kwargs)  
