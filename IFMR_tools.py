"""
Routines and classes for working with Initial-Final Mass-Relations.
"""
from itertools import pairwise
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#MS lifetime from MESA data
M_init, t_pre = np.loadtxt("data/MIST_preWD_lifetime.dat", unpack=True)
t_pre /= 1e9 #to Gyr
log_tau_fun = interp1d(np.log(M_init), np.log(t_pre), kind='linear', \
    assume_sorted=True, copy=False, bounds_error=False, fill_value='extrapolate')

def MSLT(Mi):
    """
    Main sequence lifetime (Gyr) for an initial mass (Msun). Interpolation is
    piecewise-linear in log-log space, or in otherwords a broken power-law.
    """
    return np.exp(log_tau_fun(np.log(Mi)))

class IFMR_cls(interp1d):
    """
    IFMR class for calculating final masses from initial masses and vice-versa.
    All masses in units of Msun. Keyword arguments are passed to interp1d.
    """
    def __init__(self, ifmr_x, ifmr_y, **ifmr_kw):
        ifmr_kw.update({
            'kind': 'linear',
            'assume_sorted': True,
            'bounds_error': False,
            'fill_value': 'extrapolate'
        })

        super().__init__(ifmr_x, ifmr_y, **ifmr_kw)

        ifmr_kw['copy'] = False
        self.inv = interp1d(self.y, self.x, **ifmr_kw)

        ifmr_kw['kind'] = 'next'
        dydx = np.diff(self.y, prepend=0)/np.diff(self.x, prepend=0)
        self.grad = interp1d(self.x, dydx, **ifmr_kw)
        self.inv_grad = interp1d(self.y, 1/dydx, **ifmr_kw)

    @classmethod
    def from_mc_pairs(cls, ifmr_x, mc_pairs):
        """
        Create a piecewise linear IFMR where each segment is defined
        by a gradient and y-intercept.
        """
        if len(ifmr_x) != len(mc_pairs) + 1:
            raise ValueError
        ifmr_y = np.zeros_like(ifmr_x)
        for i, ((x0, x1), (m, c)) in enumerate(zip(pairwise(ifmr_x), mc_pairs)):
            y0, y1 = m*x0+c, m*x1+c
            ifmr_y[i] = y0 if ifmr_y[i] == 0 else 0.5*(y0+ifmr_y[i])
            ifmr_y[i+1] = y1
        return cls(ifmr_x, ifmr_y)

    @classmethod
    def from_bperp_angles(cls, ifmr_x, bperp, angles):
        """
        Create a piecewise linear IFMR from a perpendicular distance and
        set of angles (as in Andrews et al. 2015)
        """
        if len(ifmr_x) != len(angles) + 1:
            raise ValueError
        phi0 = angles[0]
        y0 = bperp/np.cos(phi0) + ifmr_x[0] * np.tan(phi0)
        ifmr_y = y0 + np.cumsum(np.diff(ifmr_x)*np.tan(angles))
        ifmr_y = np.hstack([y0, ifmr_y]) 
        return cls(ifmr_x, ifmr_y)

    def __repr__(self):
        return "IFMR_cls({}, {})".format(list(self.x), list(self.y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError("subscript key must be slice object")
        xnew, ynew = self.x[key], self.y[key]
        if len(xnew) < 2:
            raise ValueError("IFMR must have at least two break points")
        return IFMR_cls(xnew, ynew)

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
        return self.y/self.x

    @property
    def mass_loss(self):
        """
        Fraction of mass lost as a function of Mi
        """
        return 1-self.Mf_Mi

    def plot(self, kind='y', *args, **kwargs):
        """
        Plot the IFMR
        """
        y_plot = getattr(self, kind)
        plt.plot(self.x, y_plot, *args, **kwargs)  
