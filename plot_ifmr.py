#!/usr/bin/env python
import numpy as np
import corner
import matplotlib.pyplot as plt
from misc import load_fitted_IFMR

BURN = -1000
PLOT_CHAINS = False
PLOT_CORNER = True
CORNER_HYPER = True
PLOT_IFMR = True
OUTLIERS = True
SIMULATED = False

if OUTLIERS and SIMULATED:
    from simulated_population import IFMR_true, TEFF_ERR, LOGG_ERR, \
        SIGMA_OUTLIER, P_outlier_true
    params_true = [P_outlier_true, SIGMA_OUTLIER, TEFF_ERR, LOGG_ERR, *IFMR_true.y]

ifmr_x, chain, lnp = load_fitted_IFMR("IFMR_MCMC_outliers_230511_extend01")
#ifmr_x, chain, lnp = load_fitted_IFMR("IFMR_MCMC_outliers_monoML_230519")
#ifmr_x, chain, lnp = load_fitted_IFMR("IFMR_MCMC_outliers_Mch_230522")

final = chain[:,BURN::5,:].reshape((-1, chain.shape[-1]))
best_coords = np.where(lnp == lnp.max())
best = chain[best_coords[0][0], best_coords[1][0]]

Nwalkers, Nstep, Ndim = chain.shape
labels = ["Teff_err", "logg_err"] \
    + [f"$y_{{{x}}}$: ${Mi:.1f}\,M_\odot$" for x, Mi in enumerate(ifmr_x, 1)]
if OUTLIERS:
    labels = ["P_outlier", "scale_outlier"] + labels

########################################
# Make figures of chains and corner plot

def chain_figure(chain, final, Ndim, Nwalkers):
    plt.figure("chains", figsize=(12, 8))
    for idim, label in enumerate(labels):
        plt.subplot(5, 4, idim+1)
        for iwalker in range(min(Nwalkers, 1000)):
            plt.plot(chain[iwalker,:,idim], 'k-', alpha=0.05)
        plt.plot(np.median(chain[:,:,idim], axis=0), 'r-')
        for pc in (16, 84):
            plt.plot(np.percentile(chain[:,:,idim], pc, axis=0), 'C1-')
        if OUTLIERS and SIMULATED:
            plt.axhline(params_true[idim], c='b', ls=':')
        plt.ylabel(label)
    plt.tight_layout()
    plt.savefig("figures/IFMR_chains.png", dpi=200)
    plt.show()

def lnprob_figure(lnp, Nwalkers):
    plt.figure("log prob", figsize=(6, 4))
    for iwalker in range(Nwalkers):
        plt.plot(lnp[iwalker,:], 'k-', alpha=0.1)
    plt.ylabel("ln(P)")
    plt.tight_layout()
    plt.show()

def IFMR_figure(final):
    final_ = final[:10000,-len(ifmr_x):]
    best_ = best[-len(ifmr_x):]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for x in ifmr_x:
        plt.axvline(x, c='C3', ls=':', alpha=0.5)
    for ifmr_y in final_:
        plt.plot(ifmr_x, ifmr_y, 'k-', alpha=0.05)
    for pc in (2.5, 16, 84, 97.5):
        plt.plot(ifmr_x, np.percentile(final_, pc, axis=0), 'C1-')
    plt.plot(ifmr_x, np.median(final_, axis=0), 'r-', lw=1.5)
    plt.plot(ifmr_x, best_, '#00FF00', lw=2, ls=':')
    if SIMULATED:
        plt.plot(IFMR_true.x, IFMR_true.y, 'b-')
    plt.plot([0, 8], [0., 8.], 'b:')
    plt.axhline(1.4, c='r', ls=':')
    plt.xlim(0, 8)
    plt.ylim(0, 2.0)
    plt.xlabel("$M_i$ [M$_\odot$]")
    plt.ylabel("$M_f$ [M$_\odot$]")

    plt.subplot(1, 2, 2)
    for x in ifmr_x:
        plt.axvline(x, c='C3', ls=':', alpha=0.5)
    for ifmr_y in final_:
        plt.plot(ifmr_x, ifmr_y/ifmr_x, 'k-', alpha=0.05)
    for pc in (2.5, 16, 84, 97.5):
        plt.plot(ifmr_x, np.percentile(final_, pc, axis=0)/ifmr_x, 'C1-')
    plt.plot(ifmr_x, np.median(final_, axis=0)/ifmr_x, 'r-', lw=1.5)
    plt.plot(ifmr_x, best_/ifmr_x, '#00FF00', lw=2, ls=':')
    if SIMULATED:
        plt.plot(IFMR_true.x, IFMR_true.Mf_Mi, 'b-')
    plt.xlim(0, 8)
    plt.ylim(0, 1.0)
    plt.xlabel("$M_i$ [M$_\odot$]")
    plt.ylabel("$M_f/M_i$")

    plt.tight_layout()
    plt.savefig("figures/IFMR.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    for idim, label in enumerate(labels):
        pcs = [np.percentile(final[:,idim], pc) for pc in (16, 50, 84)]
        args = label, pcs[1], *np.diff(pcs), best[idim]
        print("{} {:.3f}_-{:.3f}^+{:.3f} {:.3f}".format(*args))

    if PLOT_CHAINS:
        chain_figure(chain, final, Ndim, Nwalkers)
        #lnprob_figure(lnp, Nwalkers)

    if PLOT_CORNER:
        data = chain[:,BURN::1,:4] if CORNER_HYPER else chain[:,BURN::1,:]
        labels_ = labels[:4] if CORNER_HYPER else labels
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        corner.corner(data, bins=50, labels=labels_, quantiles=[0.16, 0.50, 0.84], \
            color='#333333')
        plt.savefig("figures/IFMR_corner.png", dpi=200)
        plt.show()

    if PLOT_IFMR:
        IFMR_figure(final)
