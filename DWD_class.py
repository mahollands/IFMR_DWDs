import numpy as np
import pickle
from mh.MR_relation import M_from_Teff_logg, tau_from_Teff_logg

class DWDcontainer:
    """
    Class for storing information of a Double white dwarf system
    """
    def __init__(self, name, Tg_vec, Tg_cov):
        self.name = name
        self.Tg_vec = Tg_vec
        self.Tg_cov = Tg_cov
        vecMdtau, vecMtau, JacMdtau, JacMtau = Taylor_Expand_DWD(Tg_vec)
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
        T1, T2, g1, g2 = self.Tg_vec
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

def Taylor_Expand_WD(T, g, dT=10, dg=0.001, thickness='thick'):
    """
    Pre-compute Taylor-Expansions for Masses and ages
    """
    T_vals = np.array([T, T+dT, T-dT, T, T])
    g_vals = np.array([g, g, g, g+dg, g-dg])

    M_vals = M_from_Teff_logg(T_vals, g_vals, thickness)
    M, M_p_dT , M_m_dT , M_p_dg , M_m_dg = M_vals
    dM_dT = (M_p_dT - M_m_dT)/(2*dT)
    dM_dg = (M_p_dg - M_m_dg)/(2*dg)
    Taylor_M = M, dM_dT, dM_dg

    tau_vals = tau_from_Teff_logg(T_vals, g_vals, thickness)
    tau, tau_p_dT, tau_m_dT, tau_p_dg, tau_m_dg = tau_vals
    dtau_dT = (tau_p_dT - tau_m_dT)/(2*dT)
    dtau_dg = (tau_p_dg - tau_m_dg)/(2*dg)
    Taylor_tau = tau, dtau_dT, dtau_dg

    return Taylor_M, Taylor_tau

def Taylor_Expand_DWD(Tg_vec):
    (T1, T2, g1, g2) = Tg_vec
    (M1, dM1_dT1, dM1_dg1), (tau1, dtau1_dT1, dtau1_dg1) = Taylor_Expand_WD(T1, g1)
    (M2, dM2_dT2, dM2_dg2), (tau2, dtau2_dT2, dtau2_dg2) = Taylor_Expand_WD(T2, g2)

    vecMdtau = np.array([M1, M2, tau1-tau2])
    JacMdtau = np.array([
        [dM1_dT1, 0, dM1_dg1, 0],
        [0, dM2_dT2, 0, dM2_dg2],
        [dtau1_dT1, -dtau2_dT2, dtau1_dg1, -dtau2_dg2],
    ])
    vecMtau = np.array([M1, M2, tau1, tau2])
    JacMtau = np.array([
        [dM1_dT1, 0, dM1_dg1, 0],
        [0, dM2_dT2, 0, dM2_dg2],
        [dtau1_dT1, 0, dtau1_dg1, 0],
        [0, dtau2_dT2, 0, dtau2_dg2],
    ])
    return vecMdtau, vecMtau, JacMdtau, JacMtau

def load_DWDs(use_set=None, exclude_set=None):
    if use_set is not None and exclude_set is not None:
        raise ValueError("Only one of use_set and exclude_set can be used")

    with open("DWDs_Teffs_loggs.pkl", 'rb') as F:
        DWD_dict = pickle.load(F)

    if use_set is not None:
        DWDs = [DWDcontainer(name, Tg_vec, Tg_cov) \
            for name, (Tg_vec, Tg_cov) in DWD_dict.items() \
            if name in use_set]
    elif exclude_set is not None:
        DWDs = [DWDcontainer(name, Tg_vec, Tg_cov) \
            for name, (Tg_vec, Tg_cov) in DWD_dict.items() \
            if name not in exclude_set]
    else:
        DWDs = [DWDcontainer(name, Tg_vec, Tg_cov) \
            for name, (Tg_vec, Tg_cov) in DWD_dict.items()]
    return DWDs
