# IFMR_DWDs

Scripts for fitting an Initial-Final Mass-Relation (IFMR) for a set of double white
dwarfs (DWDs) with measured Teffs and loggs.

## Components
* fit_ifmr_errors.py: Fit the IFMR with free parameters for the Teff and logg
uncertainty.
* fit_ifmr_outliers_errors.py: Fit the IFMR with free parameters for the Teff and logg,
but also the probability any point is an outlier, and the scale of the extra uncertainty
in the difference of main sequence lifetimes.
* fit_simulated_ifmr_outliers_errors.py: Same as above but fitting a simulated population
of DWDs to test the accuracy of the code.
* simulated_population.py: Generate a simulated population.
* plot_ifmy.py: plot outputs from an MCMC run.
* Mi_from_Mf.py: Given an Mf value and it's error, calculate Mi and its uncertainty
as well as for the main sequence lifetime.

