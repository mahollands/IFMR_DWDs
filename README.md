# IFMR_DWDs

Scripts for fitting an Initial-Final Mass-Relation (IFMR) for a set of double white
dwarfs (DWDs) with measured Teffs and loggs. Based on the framework first introduced
by Andrews et al. (2015), but extended to include analysis of outliers and estimation
of systematic Teff/logg uncertainties

## Components
* fit_IFMR.py: Fit a multi-segment piecewise linear IFMR to a set of DWDs.
Hyper-parameters are the fraction of outlier systems, the extra dtau variance
for outliers, and Teff/logg systematic terms.
* fit_IFMR_single.py: Fit a multi-segment piecewise linear IFMR to one DWD.
Hyper-parameters are the two initial masses for the system.
* IFMR_config.py: Contains all required variables needed to fit the IFMR.
Edit this file to fit the IFMR of your choice.
* plot_ifmr.py: plot outputs from an MCMC run.
* Mi_from_Mf.py: Given an Mf value and it's error, calculate Mi and its
uncertainty as well as for the pre-WD lifetime. Includes uncertainty
in the IFMR itself.
