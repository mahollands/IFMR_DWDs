"""
Tools for Creating a DWD container object
"""
import pickle
import numpy as np

class DWDcontainer:
    """
    Class for storing information of a double white dwarf system. This includes
    the Teffs and loggs of both components, their covariance matrix of
    measurement errors, the corresponding, masses/cooling ages, and their Jacobian
    matrix for adding systematic errors to the Teff/logg
    """
    def __init__(self, name, Tg_vec, Tg_cov, TgMtau_expansion):
        self.name = name
        self.Tg_vec = Tg_vec
        self.Tg_cov = Tg_cov
        vecMdtau, vecMtau, JacMdtau, JacMtau = TgMtau_expansion
        self.vecMdtau = vecMdtau
        self.vecMtau = vecMtau
        self.JacMdtau = JacMdtau
        self.JacMtau = JacMtau

    def __repr__(self):
        return f"DWD(name='{self.name}', ...)"

    def get_Tg_cov_systematics(self, Teff_err, logg_err):
        """
        Add systematic uncertainties to Teff-logg covariance matrix
        """
        T1, T2, *_ = self.Tg_vec
        err_syst = np.array([Teff_err*T1, Teff_err*T2, logg_err, logg_err])
        return self.Tg_cov + np.diag(err_syst**2)

    def covMdtau_systematics(self, Teff_err, logg_err):
        """
        Get M1 M2 dtau covariance matrix with Teff/logg systematic errors
        """
        Tg_cov_ = self.get_Tg_cov_systematics(Teff_err, logg_err)
        return self.JacMdtau @ Tg_cov_ @ self.JacMdtau.T

    def covMtau_systematics(self, Teff_err, logg_err):
        """
        Get M1 M2 tau1 tau2 covariance matrix with Teff/logg systematic errors
        """
        Tg_cov_ = self.get_Tg_cov_systematics(Teff_err, logg_err)
        return self.JacMtau @ Tg_cov_ @ self.JacMtau.T

    def Mdtau_samples(self, Teff_err, logg_err, N_samples=1):
        """
        Draw M1 M2 dtau samples given Teff/logg systematic errors
        """
        covMdtau = self.covMdtau_systematics(Teff_err, logg_err)
        if N_samples == 1:
            return np.random.multivariate_normal(self.vecMdtau, covMdtau)
        return np.random.multivariate_normal(self.vecMdtau, covMdtau, N_samples).T

    def Mtau_samples(self, Teff_err, logg_err, N_samples=1):
        """
        Draw M1 M2 tau1 tau2 samples given Teff/logg systematic errors
        """
        covMtau = self.covMtau_systematics(Teff_err, logg_err)
        if N_samples == 1:
            return np.random.multivariate_normal(self.vecMtau, covMtau)
        return np.random.multivariate_normal(self.vecMtau, covMtau, N_samples).T

    @property
    def M1(self):
        return self.vecMtau[0]

    @property
    def M2(self):
        return self.vecMtau[1]

    @property
    def M12(self):
        return self.vecMtau[:2]

def load_DWDs(fname="DWDs_Teffs_loggs.pkl", use_set=None, exclude_set=None):
    """
    Read in DWDs from a pickled dictionary of Teff/logg measurements. Sets
    of specific DWD names to use or exclude can be provided.
    """

    if use_set is not None and exclude_set is not None:
        raise ValueError("Only one of use_set and exclude_set can be used")

    with open(f"data/{fname}", 'rb') as F:
        DWD_dict = pickle.load(F)

    if use_set is not None:
        DWD_dict = {k: v for k, v in DWD_dict.items() if k in use_set}
    elif exclude_set is not None:
        DWD_dict = {k: v for k, v in DWD_dict.items() if k not in exclude_set}
    DWDs = [DWDcontainer(name, *Tg_vec_cov) for name, Tg_vec_cov in DWD_dict.items()]

    return DWDs
