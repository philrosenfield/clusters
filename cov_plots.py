import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

plt.style.use('presentation')

sns.set_context('paper', font_scale=2)
sns.set_style('ticks')


def _joint_plot(size=8, ratio=5, space=0.2):
    """Hack edits of sns.JointGrid so I can access it mpl style"""
    from seaborn.distributions import _freedman_diaconis_bins

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
    return ax_joint, ax_marg_x, ax_marg_y


def fake_cmds(phots):
    # phots = !! ls *noast*phot
    clp = ['darkred', 'orange', 'darkgreen', 'navy', 'purple']
    f7 = {}
    for p in phots:
        ov = '{}{}'.format(p.split('_')[0].replace('fake', ''),
                           p.split('_')[2].replace('.phot', ''))
        f7[ov] = np.loadtxt(p)

    for i, age in enumerate(['7', '8', '9']):
        ax_joint, ax_marg_x, ax_marg_y = _joint_plot()
        icov = [f for f in f7.keys() if f.startswith(age)]
        for cov_str in icov:
            m1, m2 = f7[cov_str.lower()][:, 0], f7[cov_str.lower()][:, 1]
            col = m1 - m2
            i, = [j for j, c in enumerate(cov_strs) if c in cov_str.upper()]
            inds, = np.nonzero((np.abs(m2) < 40) & (np.abs(col) < 40))
            ax_joint.plot(col[inds], m2[inds], 'o', ms=4, mec='none',
                          alpha=0.3, color=clp[i], zorder=100-i)
            cb = min(_freedman_diaconis_bins(col[inds]), 50)
            mb = min(_freedman_diaconis_bins(m2[inds]), 50)
            ax_marg_x.hist(col[inds], cb, orientation='vertical',
                           histtype='step', color=clp[i], alpha=0.4, lw=3)
            ax_marg_y.hist(m2[inds], mb, orientation='horizontal',
                           histtype='step', color=clp[i], alpha=0.4, lw=3)

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
        plt.savefig('fake_cmds{}.pdf'.format(age))


def cov_lifetimes(hb=False):
    # Plot a comparison of the H or He lifetimes vs Mass to OV=0.50
    # track_summary.dat is the screen output for running
    # padova_tracks.tracks.track as main
    tau = 'tau_H'
    if hb:
        tau = 'tau_He'
    taustr = r'\{}_{{{}}}'.format(*tau.split('_'))

    data = pd.read_table('track_summary.dat', delim_whitespace=True)
    if hb:
        intp_masses = np.arange(0.6, 2, 0.001)
        xlim = (0.6, 2)
    else:
        intp_masses = np.arange(1, 10, 0.02)
        xlim = (1.15, 5.5)

    uzs = np.unique(data['Z'])
    uovs = np.array([0.3, 0.4, 0.5, 0.6])

    linesyles = ['-', '--']
    fig, ax = plt.subplots()
    # div = make_axes_locatable(ax)
    # axt = div.append_axes("top", 1.2, pad=0.1, sharex=ax)
    clp = ['darkred', 'orange', 'darkgreen', 'navy', 'purple']
    for k, z in enumerate([uzs[0], uzs[-1]]):
        intp_dict = {}
        for ov in uovs:
            df = data[(data['Z'] == z) & (data['HB'] == 0)]
            iov = [i for i, l in enumerate(df['fname'])
                   if l.startswith('OV{:.1f}'.format(ov))]
            x = df['M'].iloc[iov]
            y = df[tau].iloc[iov] / 1e6  # Units!!

            isort = np.argsort(x)
            fx = interp1d(x.iloc[isort], y.iloc[isort],
                          bounds_error=False)
            intp_dict[ov] = fx(intp_masses)
        for i, j in enumerate(uovs):
            x_ = intp_masses
            y_ = intp_dict[j] - intp_dict[0.5]
            itxt = np.argmax(np.abs(y_))
            # axt.plot(intp_masses, intp_dict[j] * 1e6)
            if k == 0:
                ax.plot(x_, y_, linesyles[k], color=clp[i],
                        label=r'$\Lambda_c={}$'.format(j))
            else:
                ax.plot(x_, y_, linesyles[k], color=clp[i])
            # if j != 0.5:
            #     ax.text(x_[itxt], y_[itxt], 'Z={}'.format(z))
        ax.set_xlabel(r'$\rm{Mass}\ (\rm{M}_\odot)$')
        yfmt = r'$\Delta {}\ (\Lambda_c - \Lambda_{{c=0.50}})\ (\rm{{Myr}})$'
        ylab = ymft.format(taustr)
        ax.set_ylabel(ylab)
        # axt.set_yscale('log')
        plt.legend()
        ax.plot(-1, -1, '-', color='k', label='$Z=0.0005$')
        ax.plot(-1, -1, '--', color='k', label='$Z=0.01$')
        fig.subplots_adjust(left=0.15)
        ax.set_xlim(xlim)

    plt.savefig('COV_{}.png'.format(tau.split('_')[1]))


def plot_compare_tracks(ptcri_loc, tracks_dir, sandro=True, Z=0.004):
    # not sure if it's needed, but makes a pretty hrd.
    cov_strs = ['OV0.3', 'OV0.4', 'OV0.5', 'OV0.6']
    masses = [1.5, 2., 2.6, 4.]
    ts_dict = {}
    for cov_str in cov_strs:
        ts = TrackSet()
        ts.tracks_base, = fileio.get_dirs(tracks_dir,
                                          criteria='%s_Z%g_' % (cov_str, Z))
        ts.find_tracks(masses=masses, hb=False)
        ts.find_tracks(masses=masses, hb=True)
        ts._load_ptcri(ptcri_loc, search_extra=cov_str, sandro=sandro)
        ptcri_attr = ts.select_ptcri('z{}_'.format(str(Z).replace('0.', '')))
        ptcri = ts.__getattribute__(ptcri_attr)
        ts.tracks = [ptcri.load_eeps(t, sandro=sandro) for t in ts.tracks]
        ts.tracks = [ptcri.load_eeps(t, sandro=sandro) for t in ts.tracks]
        ts_dict[cov_str] = ts
    # not_so_great(ts_dict)
    cols = ['darkred', 'orange', 'darkgreen', 'navy', 'purple']
    ax = plt.subplots(figsize=(6, 12))[1]
    for i, cov_str in enumerate(cov_strs):
        for k, t in enumerate(ts_dict[cov_str].tracks):
            ind = t.sptcri[ptcri.get_ptcri_name('NEAR_ZAM')]
            if k == 0:
                label = (r'$\Lambda_c=%s$' % cov_str).replace('OV', '')
            else:
                label = None
            ax.plot(t.data[logT][ind:], t.data[logL][ind:], color=cols[i],
                    label=label)
        for t in ts_dict[cov_str].hbtracks:
            ax.plot(t.data[logT], t.data[logL], color=cols[i])

    ax.set_xlim(4.3, 3.6)
    ax.set_ylim(.85, 3.5)
    ax.set_ylabel(r'$\log L\ \rm{(L_\odot)}$', fontsize=20)
    ax.set_xlabel(r'$\log T_{\rm{eff}}\ \rm{(K)}$', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.legend(loc=0, frameon=False, fontsize=16)

    ax.text(0.01, 0.01, '$Z={}$'.format(Z), transform=ax.transAxes,
            fontsize=20)

    xs = [3.91, 4., 4.08, 4.18]
    ys = [1.27, 1.75, 2.18, 2.9]
    for x, y, m in zip(xs, ys, masses):
        ax.text(x, y, '${}M_\odot$'.format(m), fontsize=20, ha='right')

    plt.savefig('COV_HRD.png')


def not_so_great(ts_dict):
    # not that illustrative...
    cols = ['darkred', 'orange', 'darkgreen', 'navy', 'purple']
    axss = [plt.subplots(nrows=4, sharex=True, figsize=(10, 8))[1]
            for _ in range(4)]
    ycols = [logT, '', 'XHE3C', 'LOG_Pc']
    ycolls = ['$\log T_{eff}$', '$\mu_c$', '$\\rho_c$', '$\log P_c$']
    for i, cov_str in enumerate(ts_dict.keys()):
        for k, t in enumerate(ts_dict[cov_str].tracks):
            t.calc_core_mu()
            axs = axss[k]
            xdata = t.data[age] / 1e6
            for j, (ycol, ycoll) in enumerate(zip(ycols, ycolls)):
                if len(ycol) == 0:
                    ydata = t.muc  # [inds]
                else:
                    ydata = t.data[ycol]  # [inds]
                axs[j].plot(xdata, ydata, lw=3, color=cols[i],
                            label='$\Lambda_c={:.1f}$'.format(t.ALFOV))
                axs[j].set_ylabel('$%s$' % ycoll)
                axs[j].set_ylim(np.min(ydata), np.max(ydata))
                plt.legend()
            axs[i].yaxis.set_major_locator(MaxNLocator(4))

            # axs[i].xaxis.set_major_formatter(NullFormatter())
            axs[0].annotate('$M_\odot={}$'.format(t.mass), (0.45, 0.85),
                            xycoords='axes fraction', fontsize=18)


# SCRAP
#
# data = pd.read_table('all_eeps.csv.acs_wfc', delim_whitespace=True, header=0)
# lss = ['-', '--', '-.', ':']
# colors = [(0.8, 0.7254901960784313, 0.4549019607843137),
#           (0.39215686274509803, 0.7098039215686275, 0.803921568627451),
#           (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
#           (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
#           (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
#           (0.5058823529411764, 0.4470588235294118, 0.6980392156862745)]
# fig, ax = plt.subplots()
# izahb = data['iptcri'] == 11
# ishb = data['hb'] == 1
# zahb = izahb & ishb
# for i, z in enumerate(np.unique(data['Z'])):
#     for j, ov in enumerate(np.unique(data['OV'])):
#         lab = '{} {}'.format(z, ov)
#         iov = data['OV'] == ov
#         df = data[zahb & iov]
#         xcol = df['logT']
#         ycol = df['logL']
#         xcol = df['F555W'] - df['F814W']
#         ycol = df['F814W']
#         ax.plot(xcol, ycol, label=lab, color=colors[i], ls=lss[j])
#         #[ax.text(xcol.iloc[k], ycol.iloc[k], df['mass'].iloc[k])
#         # for k in range(len(df['logT']))]
#
# ax.set_xlim(ax.get_xlim()[::-1])
#
#
# ov
#
# lss = ['-', '--', '-.', ':']
# colors = ['k', (0.8, 0.7254901960784313, 0.4549019607843137),
#           (0.39215686274509803, 0.7098039215686275, 0.803921568627451),
#           (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
#           (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
#           (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
#           (0.5058823529411764, 0.4470588235294118, 0.6980392156862745)]
# fig, ax = plt.subplots()
# izahb = data['iptcri'] == 11
# ishb = data['hb'] == 1
# zahb = izahb & ishb
# for i, z in enumerate(np.unique(data['Z'])[:-1]):
#     iz = data['Z'] == z
#     df3 = data[zahb & (data['OV'] == 0.3)]
#     df5 = data[(data['iptcri'] == 11) &
#                (data['Z'] == z) & (data['OV'] == 0.5) & (data['hb'] == 1)]
#     df6 = data[(data['iptcri'] == 11) & (data['Z'] == z) &
#                (data['OV'] == 0.6) & (data['hb'] == 1)]
#     ax.plot(df5['logT'], df5['logL'], label=lab, color=colors[i])
#
#     #[ax.text(float(df['logT'].iloc[k]),
#               float(df['logL'].iloc[k]),
#               df['mass'].iloc[k]) for k in range(len(df['logT']))]
#
# ax.set_xlim(ax.get_xlim()[::-1])
