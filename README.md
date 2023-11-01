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
* plot_IFMR.py: plot outputs from an MCMC run. Use with -h or --help to list
all available options.
* Mi_from_Mf.py: Given a white dwarf mass and it's uncertainty, calculate the
initial mass and pre-WD lifetime and their uncertainties. Uses a Bayesian
approach and takes into account uncertainty in the IFMR itself. Use with -h or
--help to see all available options.
* data/DWDs_Teffs_loggs.pkl: A pickled dictionary for each of the 72 fitted
DWDs in our sample. The keys are the obversation/system names. The values are
an array of Teff/logg pairs, their covariance matrix, and pre-computed data to
convert Teffs/loggs to masses and cooling ages (in particular the Jacobians to
convert uncertainties), using data from Bedard et al. (2020). Note that
WDJ1336-1620AB appears three times as "runA", "runD", and "runAD" for an
averaged measurement. If you have your own observations, you can create your
own pickled dictionary file, or provide the information directly to the
routines/classes in DWDs.py.

## Installation
Simply clone this repository to get all available scripts and data.

## Dependencies
* python version >=3.9
* emcee (for running MCMC fits)
* corner (for plotting MCMC results)
* numpy, numba, matplotlib for maths and plotting

## Data
Shortened versions (last 100 steps) of our MCMC chains are included in the
repository. The full chains (10000 steps, up to 2GB each) can be provided
upon reasonable request.
