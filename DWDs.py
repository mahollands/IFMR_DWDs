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
    def __init__(self, name, Tg, covTg, TgMtau_expansion):
        self.name = name
        self.Tg = Tg
        self.covTg = covTg
        Mdtau, Mtau, JacMdtau, JacMtau = TgMtau_expansion
        self.Mdtau = Mdtau
        self.Mtau = Mtau
        self.JacMdtau = JacMdtau
        self.JacMtau = JacMtau

    def __repr__(self):
        return f"DWD(name='{self.name}', ...)"

    def covTg_systematics(self, Teff_err, logg_err):
        """
        Add systematic uncertainties to Teff-logg covariance matrix
        """
        T1, T2, *_ = self.Tg
        err_syst = np.array([Teff_err*T1, Teff_err*T2, logg_err, logg_err])
        return self.covTg + np.diag(err_syst**2)

    def covMdtau_systematics(self, Teff_err, logg_err):
        """
        Get M1 M2 dtau covariance matrix with Teff/logg systematic errors
        """
        covTg_ = self.covTg_systematics(Teff_err, logg_err)
        return self.JacMdtau @ covTg_ @ self.JacMdtau.T

    def covMtau_systematics(self, Teff_err, logg_err):
        """
        Get M1 M2 tau1 tau2 covariance matrix with Teff/logg systematic errors
        """
        covTg_ = self.covTg_systematics(Teff_err, logg_err)
        return self.JacMtau @ covTg_ @ self.JacMtau.T

    def Mdtau_samples(self, Teff_err, logg_err, N_samples=1):
        """
        Draw M1 M2 dtau samples given Teff/logg systematic errors
        """
        covMdtau = self.covMdtau_systematics(Teff_err, logg_err)
        if N_samples == 1:
            return np.random.multivariate_normal(self.Mdtau, covMdtau)
        return np.random.multivariate_normal(self.Mdtau, covMdtau, N_samples).T

    def Mtau_samples(self, Teff_err, logg_err, N_samples=1):
        """
        Draw M1 M2 tau1 tau2 samples given Teff/logg systematic errors
        """
        covMtau = self.covMtau_systematics(Teff_err, logg_err)
        if N_samples == 1:
            return np.random.multivariate_normal(self.Mtau, covMtau)
        return np.random.multivariate_normal(self.Mtau, covMtau, N_samples).T

    def draw_Mf_samples(self, covM, IFMR, N_samples):
        """
        Using central values for final masses and the joint covariance matrix
        calculate initial masses and the IFMR jacobian using an IFMR
        """
        Mf12 = np.random.multivariate_normal(self.M12, covM, N_samples)
        ok = (Mf12 > IFMR.y[0]) & (Mf12 < IFMR.y[-1]) #reject samples outside of IFMR
        ok = np.all(ok, axis=1)
        return Mf12[ok,:].T

    @property
    def M1(self):
        return self.Mtau[0]

    @property
    def M2(self):
        return self.Mtau[1]

    @property
    def M12(self):
        return self.Mtau[:2]

    @property
    def tau1(self):
        return self.Mtau[2]

    @property
    def tau2(self):
        return self.Mtau[3]

    @property
    def tau12(self):
        return self.Mtau[2:]

    @property
    def dtau(self):
        return self.Mdtau[2]


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
    DWDs = [DWDcontainer(name, *data) for name, data in DWD_dict.items()]

    return DWDs

##############################################################################
# Sets of DWDs to either include of exclude from the entire list

good_DWDs = {
    "220225" : {
        "runA_WDJ0855-2637AB",
        "runA_WDJ1015+0806AB",
        "runA_WDJ1019+1217AB",
        "runA_WDJ1124-1234AB",
        "runA_WDJ1254-0218AB",
        "runA_WDJ1336-1620AB",
       #"runA_WDJ1346-4630AB", #?
       #"runA_WDJ1445+2921AB", #?
        "runA_WDJ1636+0927AB",
        "runA_WDJ1856+2916AB",
        "runA_WDJ1907+0136AB",
       #"runA_WDJ1953-1019AB", #?
        "runA_WDJ2131-3459AB",
        "runA_WDJ2142+1329AB",
       #"runA_WDJ2223+2201AB", #?
        "runB_WDJ1313+2030AB",
        "runB_WDJ1338+0439AB",
        "runB_WDJ1339-5449AB",
       #"runB_WDJ1535+2125AB", #?
        "runB_WDJ1729+2916AB",
       #"runB_WDJ1831-6608AB", #?
        "runB_WDJ1859-5529AB",
       #"runB_WDJ1904-1946AB", #?
       #"runC_WDJ0007-1605AB",
        "runC_WDJ0052+1353AB",
       #"runC_WDJ0101-1629AB", #?
       #"runC_WDJ0109-1042AB", #?
        "runC_WDJ0120-1622AB",
        "runC_WDJ0215+1821AB",
        "runC_WDJ0410-1641AB",
        "runC_WDJ0510+0438AB",
        "runC_WDJ2058+1037AB",
        "runC_WDJ2122+3005AB",
        "runC_WDJ2139-1003AB",
        "runC_WDJ2242+1250AB",
    },
    "220630" : {
        "runA_WDJ0855-2637AB",
        "runA_WDJ1015+0806AB", #!
        "runA_WDJ1019+1217AB",
        "runA_WDJ1124-1234AB",
        "runA_WDJ1254-0218AB",
        "runA_WDJ1336-1620AB",
        "runA_WDJ1346-4630AB", #?
        "runA_WDJ1636+0927AB",
        "runA_WDJ1827+0403AB",
        "runA_WDJ1856+2916AB",
        "runA_WDJ1953-1019AB", #?
        "runA_WDJ2131-3459AB",
        "runA_WDJ2142+1329AB", #!
        "runB_WDJ1215+0948AB",
        "runB_WDJ1313+2030AB",
        "runB_WDJ1338+0439AB", #!
        "runB_WDJ1339-5449AB",
        "runB_WDJ1535+2125AB", #?
        "runB_WDJ1729+2916AB", #!
        "runB_WDJ1831-6608AB", #?
        "runB_WDJ1859-5529AB", #?
        "runB_WDJ1904-1946AB", #?
        "runC_WDJ0002+0733AB", #?
        "runC_WDJ0052+1353AB",
        "runC_WDJ0215+1821AB",
        "runC_WDJ0410-1641AB",
        "runC_WDJ0510+0438AB",
        "runC_WDJ2058+1037AB",
        "runC_WDJ2242+1250AB",
        "runD_WDJ0920-4127AB",
        "runD_WDJ1014+0305AB",
        "runD_WDJ1100-1600AB",
        "runD_WDJ1336-1620AB",
        "runD_WDJ1445-1459AB",
        "runD_WDJ1804-6617AB",
        "runD_WDJ1929-3000AB",
        "runD_WDJ2007-3701AB",
        "runD_WDJ2026-5020AB",
        "runD_WDJ2100-6011AB",
        "runD_WDJ2150-6218AB",
    },
}

bad_DWDs = {
    "220701" : {
        "runA_WDJ1636+0927AB", #poor fit to magnetic
        "runC_WDJ0120-1622AB", #terrible flux calibration
        "runD_WDJ1350-5025AB", #A component is a warm DC
        "runAD_WDJ1336-1620AB", #using A and D results independently
    },
    "220714" : {
        "runA_WDJ1636+0927AB", #poor fit to magnetic
        "runA_WDJ1827+0403AB", #B component v. low mass
        "runC_WDJ0120-1622AB", #terrible flux calibration
        "runC_WDJ0309+1505AB", #B component is poor fit and V. low mass
        "runC_WDJ2122+3005AB", #B component v. low mass
        "runD_WDJ1350-5025AB", #A component is a warm DC
        "runD_WDJ1211-4551AB", #B component v. low mass
        "runD_WDJ1557-3832AB", #both component v. low mass
        "runAD_WDJ1336-1620AB", #using A and D results independently
    },
    "220728" : {
        "runA_WDJ1124-1234AB", #A component has a He atm
        "runA_WDJ1636+0927AB", #poor fit to magnetic
        "runA_WDJ1827+0403AB", #B component v. low mass
        "runB_WDJ1729+2916AB", #B component has a He atm
        "runC_WDJ0120-1622AB", #terrible flux calibration
        "runC_WDJ0309+1505AB", #B component is poor fit and V. low mass
        "runC_WDJ2122+3005AB", #B component v. low mass and A componet has He atm
        "runD_WDJ1100-1600AB", #A component has a He atm
        "runD_WDJ1211-4551AB", #B component v. low mass
        "runD_WDJ1350-5025AB", #A component has a He atm or strongly magnetic
        "runD_WDJ1557-3832AB", #both component v. low mass
        "runD_WDJ1929-3000AB", #B component has a He atm
        "runAD_WDJ1336-1620AB", #using A and D results independently
    },
    "220809" : {
        "runA_WDJ1124-1234AB", #A component has a He atm
        "runA_WDJ1636+0927AB", #poor fit to magnetic
        "runA_WDJ1827+0403AB", #B component v. low mass
        "runB_WDJ1729+2916AB", #B component has a He atm
        "runC_WDJ0002+0733AB", #A component poor fit to phot due to magnetism
        "runC_WDJ2122+3005AB", #B component v. low mass and A componet has He atm
        "runD_WDJ1211-4551AB", #B component v. low mass
        "runD_WDJ1350-5025AB", #A component has a He atm or strongly magnetic
        "runD_WDJ1557-3832AB", #both components v. low mass
        "runD_WDJ1804-6617AB", #B component v. low mass
        "runD_WDJ1929-3000AB", #B component has a He atm
        "runAD_WDJ1336-1620AB", #using A and D results independently
    },
    "230206" : {
        "runA_WDJ1124-1234AB", #A component has a He atm
        "runA_WDJ1636+0927AB", #poor fit to magnetic
        "runA_WDJ1827+0403AB", #B component v. low mass
        "runB_WDJ1729+2916AB", #B component has a He atm
        "runC_WDJ0002+0733AB", #A component poor fit to phot due to magnetism
        "runC_WDJ2122+3005AB", #B component v. low mass and A componet has He atm
        "runD_WDJ1211-4551AB", #B component v. low mass
        "runD_WDJ1350-5025AB", #A component has a He atm or strongly magnetic
        "runD_WDJ1557-3832AB", #both components v. low mass
        "runD_WDJ1804-6617AB", #B component v. low mass
        "runD_WDJ1929-3000AB", #B component has a He atm
        "runE_WDJ0240-3248AB", #A component magnetic with poor phot fit
        "runE_WDJ1929-5313AB", #both components v. low mass
        "runAD_WDJ1336-1620AB", #using A and D results independently
    },
    "230511" : {
        "runA_WDJ1124-1234AB", #A component has a He atm
        "runA_WDJ1314+1732AB", #magnetism issue
        "runA_WDJ1636+0927AB", #poor fit to magnetic
        "runA_WDJ1827+0403AB", #B component v. low mass and A compt poor phot fit
        "runB_WDJ1729+2916AB", #B component has a He atm
        "runB_WDJ1859-5529AB", #poor agreement between A and B -- B possibly He atm
        "runB_WDJ2230-7513AB", #Both objects v. low mass
        "runC_WDJ0002+0733AB", #A component poor fit to phot due to magnetism
        "runC_WDJ0007-1605AB", #A component poor phot fit -- possibly He atm
        "runC_WDJ2122+3005AB", #B component v. low mass and A component has He atm
        "runD_WDJ1211-4551AB", #B component v. low mass
        "runD_WDJ1350-5025AB", #A component has a He atm or strongly magnetic
        "runD_WDJ1557-3832AB", #both components v. low mass
        "runD_WDJ1804-6617AB", #B component v. low mass
        "runD_WDJ1929-3000AB", #B component has a He atm
        "runE_WDJ0240-3248AB", #A component magnetic with poor phot fit
        "runE_WDJ1929-5313AB", #both components v. low mass
        "runE_WDJ2248-5830AB", #A component low mass
        "runE_WDJ2259+1404AB", #A component magnetic throwing off phot fit
        "runAD_WDJ1336-1620AB", #using A and D results independently
    },
    "230531" : {
        "runA_WDJ1124-1234AB", #A component has a He atm
        "runA_WDJ1314+1732AB", #magnetism issue
        "runA_WDJ1636+0927AB", #poor fit to magnetic
        "runA_WDJ1827+0403AB", #B component v. low mass and A compt poor phot fit
        "runB_WDJ1729+2916AB", #B component has a He atm
        "runB_WDJ1859-5529AB", #poor agreement between A and B -- B possibly He atm
        "runB_WDJ2230-7513AB", #Both objects v. low mass
        "runC_WDJ0002+0733AB", #A component poor fit to phot due to magnetism
        "runC_WDJ0007-1605AB", #A component poor phot fit -- possibly He atm
        "runC_WDJ2122+3005AB", #B component v. low mass and A component has He atm
        "runD_WDJ1211-4551AB", #B component v. low mass
        "runD_WDJ1350-5025AB", #A component has a He atm or strongly magnetic
        "runD_WDJ1557-3832AB", #both components v. low mass
        "runD_WDJ1804-6617AB", #B component v. low mass
        "runD_WDJ1929-3000AB", #B component has a He atm
        "runE_WDJ0240-3248AB", #A component magnetic with poor phot fit
        "runE_WDJ1929-5313AB", #both components v. low mass
        "runE_WDJ2248-5830AB", #A component low mass
        "runE_WDJ2259+1404AB", #A component magnetic throwing off phot fit
        "runA_WDJ1336-1620AB", #using combined A and D measurement instead
        "runD_WDJ1336-1620AB", #using combined A and D measurement instead
    },
}
