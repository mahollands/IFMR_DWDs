import numpy as np
import pickle
import corner
import matplotlib.pyplot as plt
from fit_ifmr_errors import ifmr_x
from mcmc_functions import MSLT, get_Mf12_dtau, loglike_Mi12_outliers
from scipy.interpolate import interp1d
from misc import Taylor_Expand_DWD

BURN = -50
PLOT_CHAINS=True
PLOT_CORNER=True
PLOT_IFMR=True
outliers=True

if outliers:
    chain = np.load("IFMR_MCMC_outliers.npy")
    lnp = np.load("IFMR_MCMC_outliers_lnprob.npy")
else:
    chain = np.load("IFMR_MCMC.npy")
    lnp = np.load("IFMR_MCMC_lnprob.npy")
final = chain[:,-1,:]

Nwalkers, Nstep, Ndim = chain.shape
labels = ["Teff_err", "logg_err"] + [f"y{x}" for x in range(1, len(ifmr_x)+1)]
if outliers:
    labels = ["P_weird", "V_weird"] + labels

########################################
# Make figures of chains and corner plot

def chain_figure(chain, Ndim, Nwalkers):
    plt.figure("chains", figsize=(12, 8))
    for idim, label in enumerate(labels):
        plt.subplot(5, 3, idim+1)
        for iwalker in range(min(Nwalkers, 1000)):
            plt.plot(chain[iwalker,:,idim], 'k-', alpha=0.05)
        plt.plot(np.median(chain[:,:,idim], axis=0), 'r-')
        for pc in (16, 84):
            plt.plot(np.percentile(chain[:,:,idim], pc, axis=0), 'C1-')
        plt.ylabel(label)
    plt.tight_layout()
    plt.savefig("IFMR_chains.png", dpi=200)
    plt.show()

def lnprob_figure(lnp, Nwalkers):
    plt.figure("log prob", figsize=(6, 4))
    for iwalker in range(Nwalkers):
        plt.plot(lnp[iwalker,:], 'k-', alpha=0.1)
    plt.ylabel("ln(P)")
    plt.tight_layout()
    plt.show()

if PLOT_CHAINS:
    chain_figure(chain, Ndim, Nwalkers)
    #lnprob_figure(lnp, Nwalkers)

if PLOT_CORNER:
    data = chain[:,BURN::5,:]
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    corner.corner(data, smooth1d=True, labels=labels, quantiles=[0.16, 0.50, 0.84])
    plt.savefig("IFMR_corner.png", dpi=200)
    plt.show()

if PLOT_IFMR:
    final_ = final[:,-len(ifmr_x):]
    for ifmr_y in final_:
        plt.plot(ifmr_x, ifmr_y, 'k-', alpha=0.05)
    plt.plot(ifmr_x, np.percentile(final_, 16, axis=0), 'C1-')
    plt.plot(ifmr_x, np.median(final_, axis=0), 'r-')
    plt.plot(ifmr_x, np.percentile(final_, 84, axis=0), 'C1-')
    plt.plot([0, 8], [0., 8.], 'b:')
    plt.axhline(1.4, c='r', ls=':')
    plt.xlim(0, 8)
    plt.ylim(0, 2.0)
    plt.xlabel("Mi [Msun]")
    plt.ylabel("Mf [Msun]")
    plt.savefig("IFMR.png", dpi=200)
    plt.show()

    for ifmr_y in final_:
        plt.plot(ifmr_x, ifmr_y/ifmr_x, 'k-', alpha=0.05)
    plt.plot(ifmr_x, np.percentile(final_, 16, axis=0)/ifmr_x, 'C1-')
    plt.plot(ifmr_x, np.median(final_, axis=0)/ifmr_x, 'r-')
    plt.plot(ifmr_x, np.percentile(final_, 84, axis=0)/ifmr_x, 'C1-')
    plt.xlim(0, 8)
    plt.ylim(0, 1.0)
    plt.xlabel("Mi [Msun]")
    plt.ylabel("Mf/Mi")
    plt.savefig("IFMR_mass_loss.png", dpi=200)
    plt.show()


with open("DWDs_Teffs_loggs.dat", 'rb') as F:
    DWDs = pickle.load(F)

DWDs1 = {name : Taylor_Expand_DWD(DWD) for name, DWD in DWDs.items()}
DWDs2 = {name : Taylor_Expand_DWD(DWD, separate_tau=True) for name, DWD in DWDs.items()}

print("Making plot")

for (name, DWD1), DWD2 in zip(DWDs1.items(), DWDs2.values()):
    plt.figure(name)
    plt.plot([0,15], [0,15], 'k:')
    P_coeval = []
    for params in final:
        if outliers:
            P_weird, V_weird, Teff_err, logg_err, *ifmr_y = params
        else:
            Teff_err, logg_err, *ifmr_y = params
        IFMR, IFMR_i = interp1d(ifmr_x, ifmr_y), interp1d(ifmr_y, ifmr_x)
        vecMtau, covMtau = get_Mf12_dtau(DWD2, Teff_err, logg_err)

        Mf1, Mf2, tau1, tau2 = np.random.multivariate_normal(vecMtau, covMtau)
        if not all(ifmr_y[0] < Mf < ifmr_y[-1] for Mf in (Mf1, Mf2)):
            continue
        Mi1, Mi2 = IFMR_i(Mf1), IFMR_i(Mf2)

        if outliers:
            vecMtau, covMtau = get_Mf12_dtau(DWD1, Teff_err, logg_err)
            logL_coeval, logL_weird = loglike_Mi12_outliers(Mi1, Mi2, vecMtau, \
                covMtau, IFMR, P_weird, V_weird, separate=True)
            logL_tot = np.logaddexp(logL_coeval, logL_weird)
            P_coeval.append(np.exp(logL_coeval-logL_tot))

        t1, t2 = MSLT(Mi1), MSLT(Mi2)
        dt = t1-t2
        plt.plot(t1+tau1, t2+tau2, 'C0.', ms=1)

        vecMtau, covMtau = get_Mf12_dtau(DWD2, 0, 0)

        Mf1, Mf2, tau1, tau2 = np.random.multivariate_normal(vecMtau, covMtau)
        if not all(ifmr_y[0] < Mf < ifmr_y[-1] for Mf in (Mf1, Mf2)):
            continue
        Mi1, Mi2 = IFMR_i(Mf1), IFMR_i(Mf2)

        t1, t2 = MSLT(Mi1), MSLT(Mi2)
        plt.plot(t1+tau1, t2+tau2, 'C1.', ms=1)

    plt.loglog()
    plt.xlim(0.2, 13)
    plt.ylim(0.2, 13)
    tick_labels = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    plt.xticks(tick_labels, tick_labels)
    plt.yticks(tick_labels, tick_labels)
    plt.xlabel("t_tot1 [Gyr]")
    plt.ylabel("t_tot2 [Gyr]")
    if outliers:
        plt.title("$P_\mathrm{{coeval}}$ = {:.3f}".format(np.mean(P_coeval)))
    plt.tight_layout()
    plt.savefig(f"age_plots/{name}.png", dpi=200)
    plt.close()
    print(name)
