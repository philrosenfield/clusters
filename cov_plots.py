import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from seaborn.distributions import _freedman_diaconis_bins

plt.style.use('presentation')

sns.set_context('paper', font_scale=2)
sns.set_style('ticks')
sns.set_style('whitegrid')
sd = sns.axes_style()
sd['text.usetex'] = True
sns.set(sd)

EXT = '.pdf'
# EXT = '.png'


def _joint_plot(size=8, ratio=5, space=0.2):
    """Hack edits of sns.JointGrid so I can access it mpl style"""
    f = plt.figure(figsize=(size, size))
    gs = plt.GridSpec(ratio + 1, ratio + 1)
    ax_joint = f.add_subplot(gs[1:, :-1])
    ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
    ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)
    for ax in [ax_marg_x, ax_marg_y]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.yaxis.grid(False)
    for ax in [ax_joint, ax_marg_y, ax_marg_x]:
        ax.tick_params(top='off', right='off')
    return f, ax_joint, ax_marg_x, ax_marg_y


def fake_cmds(phots, space=0.2):
    # phots = !! ls *noast*phot
    clp = ['darkred', 'orange', 'darkgreen', 'navy', 'purple']
    cov_strs = ['OV0.3', 'OV0.4', 'OV0.5', 'OV0.6']
    f7 = {}
    f, ax_joint, ax_marg_x, ax_marg_y = _joint_plot()
    for p in phots:
        icov, = [j for j, c in enumerate(cov_strs) if c.lower() in p]
        m1, m2 = np.loadtxt(p, unpack=True)
        col = m1 - m2
        inds, = np.nonzero((np.abs(m2) < 40) & (np.abs(col) < 40))
        ax_joint.plot(col[inds], m2[inds], 'o', ms=4, mec='none',
                      alpha=0.3, color=clp[icov], zorder=100-icov)
        cb = min(_freedman_diaconis_bins(col[inds]), 50)
        mb = min(_freedman_diaconis_bins(m2[inds]), 50)

        ax_marg_x.hist(col[inds], cb, orientation='vertical',
                       histtype='step', color=clp[icov], alpha=0.4, lw=3)
        ax_marg_y.hist(m2[inds], mb, orientation='horizontal',
                       histtype='step', color=clp[icov], alpha=0.4, lw=3)

    ax_marg_x.set_yscale('log')
    ax_marg_y.set_xscale('log')
    labels = [r'$\Lambda_c=%.1f$' % float(c.replace('OV', ''))
              for c in cov_strs]
    for i in range(len(labels)):
        ax_joint.plot(100, 100, 'o', color=clp[i], label=labels[i])
    ax_joint.legend(loc='best')
    ax_joint.set_ylim(22, 10)
    ax_joint.set_xlim(-0.5, 1.9)
    ax_joint.set_xlabel('$F555W-F814W$')
    ax_joint.set_ylabel('$F814W$')
    f.tight_layout()
    f.subplots_adjust(hspace=space, wspace=space)
    # plt.savefig('fake_cmds{}.pdf'.format(age))
    return f, ax_joint, ax_marg_x, ax_marg_y


def setup_covlifetimes(data, hb=False, agescale=1e6):
    intp_dict = {}
    for ov in np.array([0.3, 0.4, 0.5, 0.6]):
        df = data[(data['Z'] == z) & (data['HB'] == 0)]
        if hb:
            dfhb = data[(data['Z'] == z) & (data['HB'] == 1)]
            dfms = df[df.M > dfhb.M.max()]
            df = dfhb.append(dfms)
        iov = [i for i, l in enumerate(df['fname'])
               if l.startswith('OV{:.1f}'.format(ov))]
        x = df['M'].iloc[iov]
        y = df[tau].iloc[iov] / agescale  # Units!!

        isort = np.argsort(x)
        fx = interp1d(x.iloc[isort], y.iloc[isort],
                      bounds_error=False)
        intp_dict[ov] = fx(intp_masses)
    return intp_dict


def cov_masslifetimes(hb=False, both=False):
    # Not used.
    # Plot a comparison of the H or He lifetimes vs Mass to OV=0.50
    # track_summary.dat is the screen output for running
    # padova_tracks.tracks.track as main
    tau = 'tau_H'
    if hb:
        tau = 'tau_He'
    taustr = r'\{}_{{{}}}'.format(*tau.split('_'))

    data = pd.read_table('track_summary.dat', delim_whitespace=True)

    intp_masses = np.arange(1, 3, 0.02)
    uzs = np.unique(data['Z'])
    uovs = np.array([0.3, 0.4, 0.5, 0.6])

    # xlim = (1.15, 5.5)
    linesyles = [':', '--', '-']
    fig, ax = plt.subplots()
    clp = ['darkred', 'orange', 'darkgreen', 'navy', 'purple']
    for k, z in enumerate([uzs[0], uzs[-1]]):
        intp_dict = setup_covlifetimes(data, hb=False, agescale=1e9)
        for i, j in enumerate(uovs):
            y_ = intp_masses
            x_ = intp_dict[j]
            ax.plot(x_, y_, linesyles[k], color=clp[i],
                    label=r'$\Lambda_c={} Z={}$'.format(j, z))

        ax.set_xlabel(r'$\rm{Age}\ (\rm{Gyr})$')
        ax.set_ylabel(r'$\rm{Mass}\ (\rm{M}_\odot)$')
        # axt.set_yscale('log')
    # stopped here...


def cov_complifetimes(hb=False, both=False):
    # Plot a comparison of the H or He lifetimes vs Mass to OV=0.50
    # track_summary.dat is the screen output for running
    # padova_tracks.tracks.track as main
    tau = 'tau_H'
    if hb:
        tau = 'tau_He'
    taustr = r'\{}_{{{}}}'.format(*tau.split('_'))

    data = pd.read_table('track_summary.dat', delim_whitespace=True)

    intp_masses = np.arange(1, 6, 0.02)
    uzs = np.unique(data['Z'])
    uovs = np.array([0.3, 0.4, 0.5, 0.6])

    xlim = (1.15, 5.5)
    linesyles = ['-', '--']
    fig, ax = plt.subplots()
    clp = ['darkred', 'orange', 'darkgreen', 'navy', 'purple']
    for k, z in enumerate([uzs[0], uzs[-1]]):
        intp_dict = setup_covlifetimes(data, hb=False)
        for i, j in enumerate(uovs):
            x_ = intp_masses
            y_ = intp_dict[j] - intp_dict[0.5]
            if k == 0:
                ax.plot(x_, y_, linesyles[k], color=clp[i],
                        label=r'$\Lambda_c={}$'.format(j))
            else:
                ax.plot(x_, y_, linesyles[k], color=clp[i])

        ax.set_xlabel(r'$\rm{Mass}\ (\rm{M}_\odot)$')
        yfmt = r'$\Delta {}\ (\Lambda_c - \Lambda_{{c=0.50}})\ (\rm{{Myr}})$'
        ylab = yfmt.format(taustr)
        ax.set_ylabel(ylab)
        # axt.set_yscale('log')

    ax.plot(-1, -1, '-', color='k', label='$Z=0.0005$')
    ax.plot(-1, -1, '--', color='k', label='$Z=0.01$')
    fig.subplots_adjust(left=0.15)
    plt.tick_params(top='off', right='off')
    ax.set_xlim(xlim)
    if both:
        plt.legend(loc=0)
    plt.savefig('COV_{}'.format(tau.split('_')[1]) + EXT)
    if both:
        hb2 = not hb
        ax2 = cov_lifetimes(hb=hb2, both=False)
        ax = [ax, ax2]
    return ax


def plot_compare_tracks(Z=0.004, cmd=False):
    # not sure if it's needed, but makes a pretty hrd.
    plt.rcParams['lines.linewidth'] -= 1
    from .data_plots import adjust_zoomgrid, setup_zoomgrid
    from . import fileio
    import os
    from padova_tracks.tracks.track import Track
    cov_strs = ['OV0.3', 'OV0.4', 'OV0.5', 'OV0.6']
    masses = [1.5, 2.]
    # not_so_great(ts_dict)
    cols = ['darkred', 'orange', 'darkgreen', 'navy', 'purple']
    tracks = []
    for cov in cov_strs:
        for mass in masses:
            ext = 'dat'
            if cmd:
                ext = 'acs_wfc'
            track_dir, = fileio.get_dirs(os.getcwd(),
                                         criteria='%s_Z%g_' % (cov, Z))
            track_name = fileio.get_files(track_dir,
                                          '*M{:.2f}*{:s}'.format(mass, ext))
            track = Track(track_name[0], match=True)
            if len(track_name) > 1:
                if cov == 'OV0.3':
                    if mass == 1.5:
                        tracks.append(Track(track_name[1], match=True))
                else:
                    # slap the PMS and HB tracks together...
                    # this won't add MSTO age to HB, only for HRD plotting.
                    df1 = pd.DataFrame(track.data)
                    track.data = df1.append(pd.DataFrame(Track(track_name[1],
                                                         match=True).data))
            tracks.append(track)
    fig, axs = setup_zoomgrid()
    for k, t in enumerate(tracks[::-1]):
        if not cmd:
            x = t.data['logT']
            y = t.data['logL']
        else:
            x = t.data['F475W'] - t.data['F814W']
            y = t.data['F814W']
        cov_str = t.base.split('OV')[1].split('_')[0]
        label = (r'$\Lambda_c=%s$' % cov_str).replace('OV', '')
        icol, = [i for i, c in enumerate(cov_strs) if cov_str in c]
        if t.mass == masses[0]:
            ax = axs[2]
            if not t.hb:
                axs[0].plot(x[200:], y[200:], label=label, color=cols[icol])
            else:
                axs[0].plot(x[200:], y[200:], color=cols[icol])
        else:
            axs[0].plot(x[200:], y[200:], color=cols[icol])
            ax = axs[1]

        if t.hb:
            ax.plot(x, y, color=cols[icol])
        else:
            ax.plot(x[1130:], y[1130:], color=cols[icol])

    if not cmd:
        zoom2_kw = {'ylim': [1.75, 2.],
                    'xlim': [3.706, 3.685]}
        zoom1_kw = {'ylim': [1.5, 2.1],
                    'xlim': [3.725, 3.685]}

        axs[0].set_xlim(4.04, 3.66)
        axs[0].set_ylim(0.8, 2.5)

        axs[0].set_ylabel(r'$\log L\ \rm{(L_\odot)}$', fontsize=20)
        axs[0].set_xlabel(r'$\log T_{\rm{eff}}\ \rm{(K)}$', fontsize=20)
        xs = [3.9, 4.]
        ys = [1.27, 1.75]
        reversey = False
    else:
        axs[0].set_ylim(2.6, -1.5)
        axs[0].set_xlim(-0.15, 1.75)

        axs[0].set_ylabel(r'$\rm{F475W}$', fontsize=20)
        axs[0].set_xlabel(r'$\rm{F475W-F814W}$', fontsize=20)
        zoom2_kw = {'ylim': [-0.3, -1],
                    'xlim': [1.4, 1.65]}
        zoom1_kw = {'ylim': [0.2, -1],
                    'xlim': [1.28, 1.5]}
        xs = [0.6, 0.1]
        ys = [1.1, 0.1]
        reversey = True

    adjust_zoomgrid(axs[0], axs[1], axs[2], zoom1_kw=zoom1_kw,
                    zoom2_kw=zoom2_kw, reversey=reversey)
    axs[1].locator_params('x', nbins=3)
    axs[2].locator_params('x', nbins=3)

    axs[0].tick_params(labelsize=16)
    axs[0].legend(loc=0, frameon=False, fontsize=16)

    axs[1].set_title('$Z={}$'.format(Z), fontsize=20)
    for ax, m in zip([axs[2], axs[1]], masses):
        ax.text(0.9, 0.01, '${}M_\odot$'.format(m), transform=ax.transAxes,
                fontsize=20, ha='right')

    for x, y, m in zip(xs, ys, masses):
        axs[0].text(x, y, '${}M_\odot$'.format(m), fontsize=20, ha='center')
    outfile = 'COV_HRD'
    if cmd:
        outfile = 'COV_CMD'
    plt.savefig(outfile + EXT)

plot_compare_tracks(Z=0.006, cmd=True)
plot_compare_tracks(Z=0.006)
