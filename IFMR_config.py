"""
Variables in this file can be edited for different setup options to fit the
IFMR. Then simply run the script 'fit_IFMR.py'.
"""

#######################################
# Intial-mass grid for fitting the IFMR
ifmr_x = [0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8]

#######################################
# Switches for the IFMR Bayesian model
MONOTONIC_IFMR = True
MONOTONIC_MASS_LOSS = False
MCH_PRIOR = False
STRICT_MASS_LOSS = True
MI_PRIOR = True
TAU_PRIOR = True
N_MARGINALISE = 1600

######################################
# MCMC parameters
N_CPU = 10
Nwalkers = 100
Nstep = 10000

#####################################
# input/output filenames
f_MCMC_out = "IFMR_MCMC_230605"
f_continue_from = "" #pick up fit from this attempt (ignore if empty)

#####################################
# Teff/logg systematic error sums as
# measured from WDJ1336-1620AB
S_T = 1.577E-4
S_g = 2.978E-3

#####################################
# DWD selection
f_DWDs = "DWDs_Teffs_loggs.pkl"
exclude_set = "230531" #See DWDs.py for choices
