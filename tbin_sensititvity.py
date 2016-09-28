import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from match.scripts.sfh import SFH

# from .config import FIGEXT


sns.set_context('paper', font_scale=2)
sns.set_style('whitegrid')
sd = sns.axes_style()
sd['text.usetex'] = True
sns.set(sd)
# /Users/rosenfield/Dropbox (tehom)/NGC1978/inner500/tbin_sensitivity'
os.chdir("/Users/rosenfield/Dropbox (tehom)/NGC1978/inner500/tbin_sensitivity")
sfhfiles = glob.glob('*csp*sfh')
sfhs = [SFH(s) for s in sfhfiles]

# true20 = [9.1511331, 9.1572233, 4.0e-1, 20]
true50 = [9.13552311, 9.1511331, 1.800e-1, 50]
true100 = [9.1511331, 9.18076445, 1.000e-1, 100]
true500 = [9.11933105, 9.25917031, 2.00e-2, 500]

# sfh20 = sfhs[0:3]
sfh50 = [s for s in sfhs if '050Myr' in s.name]
sfh100 = [s for s in sfhs if '100Myr' in s.name]
sfh500 = [s for s in sfhs if '500Myr' in s.name]

dt50 = [s for s in sfhs if '5e+07' in s.name]
dt100 = [s for s in sfhs if '1e+08' in s.name]
dt500 = [s for s in sfhs if '5e+08' in s.name]

plotset = [sfh50, sfh100, sfh500]

fig, axs = plt.subplots(nrows=3, figsize=(8, 8), sharex=True)
axs = axs.ravel()
for i, (sfhi, truei) in enumerate(zip(plotset, [true50, true100, true500])):
    verts = np.array([[8.5, 0],
                      [truei[0], 0],
                      [truei[0], truei[2]],
                      [truei[1], truei[2]],
                      [truei[1], 0],
                      [9.5, 0]])
    axs[i].plot(verts[:, 0], verts[:, 1], color='k', alpha=0.3, lw=6)
    # sf = (10 ** truei[1] - 10 ** truei[0]) * truei[2] / 1e6

    for sfh in sfhi:
        lage, sfr = sfh.plot_bins()
        tm = float(sfh.name.split('_')[2].replace('dt', '')) / 1e6
        axs[i].plot(lage, sfr, label=r'$\Delta t={}$'.format(tm))

        axs[i].grid('off')

        axs[i].text(0.05, 0.85,
                    r'${{\rm Input}}\ \Delta t={}$'.format(truei[-1]),
                    transform=axs[i].transAxes, fontsize=18)

axs[0].legend(loc='best')
axs[0].set_xlim(8.8, 9.4)
axs[1].set_ylabel(r'${\rm SFR\ (M}_\odot/{\rm yr)}$', fontsize=20)
axs[-1].set_xlabel(r'$\log\ {\rm Age}$', fontsize=20)
plt.savefig('tbin_sensitivity{}'.format('.pdf'))
