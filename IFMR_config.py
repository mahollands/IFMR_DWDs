"""
Variables in this file can be edited for different choices of fit
"""

#######################################
# Intial mass grid for fitting the IFMR
ifmr_x = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8]

#######################################
# Switches for the IFMR model
MONOTONIC_IFMR = True
MONOTONIC_MASS_LOSS = False
MCH_PRIOR = False
STRICT_MASS_LOSS = True
N_MARGINALISE = 10000

######################################
# MCMC parameters
N_CPU = 10
Nwalkers = 1000
Nstep = 10000

#####################################
# filenames
f_MCMC_out = "IFMR_MCMC_230601"
f_continue_from = "" #pick up fit from this attempt

#####################################
# Teff/logg systematic error sums as
# measured from WDJ1336-1620AB
S_T = 1.577E-4
S_g = 2.978E-3
