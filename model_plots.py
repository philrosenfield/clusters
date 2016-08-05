'''
Plots for paper that use models primarily
'''
import padova_tracks
import match.scripts as match
from ResolvedStellarPops import fileio, graphics
import numpy as np
import matplotlib.pylab as plt
import os

import pandas as pd

td = padova_tracks.TrackDiag()
# cov_strs = ['OV0.3', 'OV0.4', 'OV0.5', 'OV0.6', 'OV0.7']
cov_strs = ['OV0.3', 'OV0.4', 'OV0.6']
tracks_dir = '/Users/rosenfield/research/stel_evo/tracks/CAF09_MC_S15/tracks/'
ptcri_loc = '/Users/rosenfield/research/stel_evo/tracks/CAF09_MC_S15/data/'


csv = 'combined_files.csv'
data = pd.read_csv(csv)
inds, = np.nonzero(np.exp(.5 * (data['fit'].min() - data['fit'])))
cols = [c for c in df.columns if len(np.unique(df[c])) > 1 and
        c != 'sfr' and c != 'fit' and c != 'Unnamed: 0']

for i in range(len(cols)):
    for j in range(len(cols)):
        if i <= j:
            continue
        if len(np.unique(df[cols[i]])) > 1 and \
           len(np.unique(df[cols[j]])) > 1:
            ax = pcolor_(df[cols[i]], df[cols[j]], df.fit)
            ax.set_ylabel(cols[j])
            ax.set_xlabel(cols[i])

fig, axs = plt.subplots(nrows=3, ncols=3)
for i, j in itertools.product(range(3), range(3)):
        if i == j:
            pass
        else:
            pcolor_(data[cols[j]], data[cols[i]], data['fit'], ax=axs[i, j])
            axs[i, j].set_xlabel(cols[j])
            axs[i, j].set_ylabel(cols[i])

        axs[i, j].set_visible(False)


# SCRAP
def plot_compare_at_eep(Z=0.004, comp='OV0.5', hb=False, eep_name='POINT_C',
                        xattr='MASS', yattr='AGE', yfunc='1/1e6*',
                        lab_name=None, sandro=True):
    """
    Plot interpolated values across all tracks for each track set of the cov
    grid.

    Using all defaults, will plot the interpolated MSTO age as a function of
    mass, (should set lab_name='MSTO' or set sandro=False and eep_name='MSTO').

    Notes
    -----
    File locations are hard coded

    Has a nice TrackSet loop that calls TrackSet.relationships that could
    be taken out to a more general function

    Parameters
    ----------
    Z : float [0.004]
        metallicity

    comp : str [OV0.5]
        OV%.1f core overshoot value from the track directory name as the base
        of comparison

    hb : bool [False]
        toggle if these are horizontal branch tracks
        (will have to adjust xlim, ylim)

    eep_name : str ['POINT_C']
        eep to compare all tracks must follow sandro's or Phil's definitions
        (sandro True/False)

    xattr, yattr : str ['Mass'], str ['AGE']
        columns of the track file to interpolate

    xfunc, yfunc : str [None], str ['1/1e6*']
        eval eg, '%s(xdata)' % xfunc Myr: '1 / 1e6 *' log: 'np.log10'

    lab_name : str [None]
        name for the yaxis label, takes eep_name.title() if None

    sandro : bool
        toggle use of Sandro's EEPs or Phil's

    Returns
    -------
    ax : matplotlib.axes object
    """
    if hb:
        mass_arr = np.arange(1., 1.8, 0.01)
    else:
        mass_arr = np.arange(0.2, 4.1, 0.1)
    tsearch = '*F7_*PMS'
    if hb:
        tsearch += '.HB'
    ts_dict = {}
    for cov_str in cov_strs:
        track_base, = fileio.get_dirs(tracks_dir,
                                      criteria='%s_Z%g_' % (cov_str, Z))
        track_names = fileio.get_files(track_base, tsearch)
        ts = padova_tracks.TrackSet()
        ts.tracks = [padova_tracks.Track(t) for t in track_names]
        ts.relationships(eep_name, xattr, yattr, yfunc=yfunc, hb=hb,
                         sandro=sandro, ptcri_loc=ptcri_loc,
                         ptcri_search_extra=cov_str)
        lage_arr = ts.mass_age_interp(mass_arr)
        ts_dict[cov_str] = lage_arr

    colors = ['darkred', 'orange', 'green', 'navy', 'purple']
    ax = plt.subplots()[1]

    for i, cov_str in enumerate(cov_strs):
        if cov_str == comp:
            continue

        cov_comp = ts_dict[cov_str] - ts_dict[comp]
        label = (r'$\Lambda_c=%s$' % cov_str).replace('OV', '')
        ax.plot(mass_arr, cov_comp, lw=5, color='k')
        ax.plot(mass_arr, cov_comp, lw=3, label=label, color=colors[i])

    if lab_name is None:
        lab_name = eep_name.title()
    cov = float(comp.replace('OV', ''))
    ylab = r'$\tau_{%s, \Lambda_c = x} - \tau_{%s, \Lambda_c=%.1f}\ (\rm{Myr})$' % (lab_name, lab_name, cov)
    ax.set_ylabel(ylab, fontsize=20)
    ax.set_xlabel(r'$\rm{%s}\ M_\odot$' % xattr.title(), fontsize=20)
    # ax.set_xlim(0.9, 3.9)
    # ax.set_ylim(-200, 200)
    ax.tick_params(labelsize=16)
    ax.legend(loc=0, frameon=False, fontsize=16)
    outfig = 'comp_lambdac_z%g.png' % Z
    plt.savefig(outfig)
    return ax


def cov_cluster_grid_plots(ss='sfh',
                           base='/Users/phil/research/clusters/n419/match'):
    sfh_files = fileio.get_files(base, '*' + ss)
    msfhs = [match.utils.MatchSFH(s) for s in sfh_files]
    nsfhs = len(msfhs)
    labs = [m.name.split('.')[0].replace('_', r'\ ') for m in msfhs]
    bestfits = [msfhs[i].bestfit for i in range(len(msfhs))]

    ylabs = [r'$SFR\ \rm{(M_\odot/yr)}$', r'$Z$']
    for j, val in enumerate(['SFR', 'MH']):
        axs = plt.subplots(nrows=nsfhs, sharex=True, figsize=(8, 12))[1]
        for i, msfh in enumerate(msfhs):
            lage, yval = msfh.plot_bins(val=val.lower())
            lw = 2
            if val == 'MH':
                yval = 0.02 * 10 ** yval
                yval[yval == 0.02] = np.nan
            axs[i].plot(lage, yval, lw=lw, color='k')
            axs[i].set_title(r'$\rm{%s}: %.4g$' % (labs[i], bestfits[i]))
        axs[0].set_xlim(8.6, 9.4)
        axs[-1].set_xlabel(r'$\log Age\ \rm{(yr)}$', fontsize=20)
        axs[3].set_ylabel(ylabs[j], fontsize=20)
        plt.savefig('n419_covgrid_%s_lage.png' % val.lower())


def match_diagnostic_plots(base=os.getcwd(), sfh_str='*cov?.sfh',
                           cmd_str='*cov?.*cmd',
                           filter1=r'F555W', filter2=r'F814W\ {\rm (HRC)}',
                           labels='default'):
    from matplotlib.ticker import NullFormatter

    cmd_files = fileio.get_files(base, cmd_str)
    cmd_files = sorted(cmd_files, key=lambda x: x[-9])
    sfh_files = fileio.get_files(base, sfh_str)
    sfh_files = sorted(sfh_files, key=lambda x: x[-5])

    msfhs = [match.utils.MatchSFH(s) for s in sfh_files]

    cols = ['darkred', 'orange', 'black', 'navy']
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    for i, msfh in enumerate(msfhs):
        lab = msfh.name.split('_')[1].replace('.sfh', '$').replace('cov', r'$\Lambda_c=0.')
        plt_kw = {'color': cols[i], 'label': lab}
        msfh.age_plot(ax=ax1, plt_kw=plt_kw)
        msfh.age_plot(val='mh', convertz=True, ax=ax2, plt_kw=plt_kw)
        ax1.xaxis.set_major_formatter(NullFormatter())
        plt.subplots_adjust(hspace=0.1)
    plt.legend(loc=0, frameon=False)
    plt.savefig('n419_cov_match.png')

    if labels == 'default':
        labels = [r'${\rm %s}$' % i for i in ('data', 'model', 'diff', 'sig')]

    for i, cmd_file in enumerate(cmd_files):
        labels[1] = '${\\rm %s}$' % msfhs[i].name.split('.')[0].replace('_', '\ ')
        labels[-1] = '$%.1f$' % msfhs[i].bestfit
        match.graphics.pgcmd(cmd_file, filter1=filter1, filter2=filter2,
                             labels=labels, figname=cmd_file + '.png')
        match.utils.match_stats(sfh_files[i], cmd_file, nfp_nonsfr=5,
                                nmc_runs=10000, outfile=cmd_file+'.dat')

    ssp_files = fileio.get_files(base, '*ssp*scrn')
    if len(ssp_files) > 0:
        [match.utils.strip_header(ssp) for ssp in ssp_files]

    return


def cov_testgrid_plots(ss='zc'):
    '''make SFR vs Log Age, Z vs Log Age, and Best fit vs COV plots'''
    # base = '/Users/phil/research/clusters/n419/match/fake_tests'
    base = '/home/rosenfield/research/clusters/n419/match/fake/fake_9.00_9.20_1.44_0.004_0.05'

    sfh_files = fileio.get_files(base, '*' + ss)
    msfhs = [match.utils.MatchSFH(s) for s in sfh_files]

    for i in range(len(msfhs)):
        # assuming n419_s12_cov7_c4in format
        msfhs[i].cov_in = \
            float(msfhs[i].name.split('_c')[-2].replace('ov', '')) * .1
        msfhs[i].cov = \
            float(msfhs[i].name.split('_c')[-1].replace('in.%s' % ss, '')) * .1

    cov_ins = [msfhs[i].cov_in for i in range(len(msfhs))]
    ucovs = np.unique(cov_ins)
    ncovs = len(ucovs)
    cols = graphics.discrete_colors(ncovs)

    ylabs = [r'$SFR\ \rm{(M_\odot/yr)}$', r'$Z$']
    for i, val in enumerate(['SFR', 'MH']):
        axs = plt.subplots(nrows=ncovs, sharex=True, figsize=(8, 12))[1]
        for msfh in msfhs:
            ind_cc = list(ucovs).index(msfh.cov)
            ind_cf = list(ucovs).index(msfh.cov_in)
            lage, yval = msfh.plot_bins(val=val.lower())
            lw = 2
            if msfh.cov == msfh.cov_in:
                lw = 4
            if val == 'MH':
                yval = 0.02 * 10 ** yval
                yval[yval == 0.02] = np.nan
            axs[ind_cc].plot(lage, yval, lw=lw,
                             label=r'$\Lambda_{cf}=%.1f$' % (msfh.cov_in),
                             color=cols[ind_cf])

            axs[ind_cc].set_title(r'$\Lambda_{cc}=%.1f$' % msfh.cov)
        _ = [ax.legend(loc=0, frameon=False) for ax in axs]
        axs[0].set_xlim(8.8, 9.4)
        axs[-1].set_xlabel(r'$\log Age\ \rm{(yr)}$', fontsize=20)
        axs[2].set_ylabel(ylabs[i], fontsize=20)
        plt.savefig(os.path.join(base,
                                 'cov_testgrid_%s_lage.png' % val.lower()))

    ibins = np.digitize(cov_ins, bins=ucovs) - 1
    bestfits = [msfhs[i].bestfit for i in range(len(msfhs))]

    ax = plt.subplots()[1]
    for i in range(ncovs):
        for k, j in enumerate(np.nonzero(ibins == i)[0]):
            ax.plot(cov_ins[j], bestfits[j], 'o', ms=15, color=cols[k])

    _ = [ax.plot(-99, yval[0], 'o', label=r'$\Lambda_{cc}=%.1f$' % ucovs[i],
                 color=cols[i])
         for i in range(ncovs)]
    ax.set_xlim(0.29, 0.71)
    #ax.set_ylim(10000, 70000)
    ax.set_ylabel(r'$\rm{Prob.}$', fontsize=20)
    ax.set_xlabel(r'$\Lambda_c\ \rm{in\ fake}$', fontsize=20)
    ax.legend(loc=0, frameon=False, numpoints=1)
    plt.savefig(os.path.join(base, 'cov_testgrid_prob_cov.png'))


def compare_khd(tracks, fusion=True, convection=True,
                khd_dict={'CONV':  'black'}):
    cols = ['darkred', 'orange', 'darkgreen', 'navy', 'purple']
    axss = [plt.subplots(nrows=4, sharex=True, figsize=(10, 8))[1]
            for _ in range(4)]
    ycols = [logT, '', 'LOG_RHc', 'LOG_Pc']
    ycolls = ['$\log T_{eff}$', '$\mu_c$', '$\\rho_c$', '$\log P_c$']
    for i, cov_str in enumerate(cov_strs):
        for k, t in enumerate(ts_dict[cov_str].tracks):
            axs = axss[k]
            xdata = t.data[age] / 1e6
            for j, (ycol, ycoll) in enumerate(zip(ycols, ycolls)):
                if len(ycol) == 0:
                    ydata = t.muc  # [inds]
                else:
                    ydata = t.data[ycol]  # [inds]
                axs[j].plot(xdata, ydata, lw=3, colors=cols[k],
                            label='$\Lambda_c={:.1f}$'.format(t.ALFOV))
                axs[j].set_ylabel('$%s$' % ycoll)
                axs[j].set_ylim(np.min(ydata), np.max(ydata))
                plt.legend()
            axs[i].yaxis.set_major_locator(MaxNLocator(4))

            # axs[i].xaxis.set_major_formatter(NullFormatter())
            axs[0].annotate('$\Lambda_c=%.1f$' % t.ALFOV, (0.45, 0.85),
                            xycoords='axes fraction', fontsize=18)
            axs = axss[k]
            t._load_iptcri(t.ptcri_file)
            axs[i].plot(t.data[])
            td.kippenhahn(t, heb_only=False, between_ptcris=[4, 6],
                          xscale='linear', khd_dict=khd_dict,
                          fusion=fusion, four_tops=True)
            axs[i].vlines(t.data[age][t.iptcri[5]] / 1e6, 0, 1,
                          color='grey', lw=2)
            axs[i].vlines(t.data[age][t.iptcri[10]] / 1e6, 0, 1,
                          color='grey', lw=2)
            axs[i].annotate('$\Lambda_c=%.1f$' % t.ALFOV, (0.45, 0.85),
                            xycoords='axes fraction', fontsize=18)
            # fig.subplots_adjust(left=0.08, right=0.95, top=0.95, hspace=0.4)

    return axs


def compare_covs_khd():
    base = '/Users/phil/research/stel_evo/CAF09_D13/'
    cov = [0.3, 0.5, 0.7]

    track_names = \
        [base +
         'tracks/MC_S13_OV0.3_Z0.004_Y0.2557/Z0.004Y0.2557OUTA1.74_F7_M2.40.PMS',
         base + \
         'tracks/MC_S13_OV0.5_Z0.004_Y0.2557/Z0.004Y0.2557OUTA1.74_F7_M2.40.PMS',
         base + \
         'tracks/MC_S13_OV0.7_Z0.004_Y0.2557/Z0.004Y0.2557OUTA1.74_F7_M2.40.PMS']
    ptcri_names = [base + 'data/p2m_ptcri_CAF09_D13_MC_S13_OV0.3_Z0.004_Y0.2557.dat',
                   base + 'data/p2m_ptcri_CAF09_D13_MC_S13_OV0.5_Z0.004_Y0.2557.dat',
                   base + 'data/p2m_ptcri_CAF09_D13_MC_S13_OV0.7_Z0.004_Y0.2557.dat']
    tracks = [padova_tracks.Track(t) for t in track_names]
    for i in range(len(tracks)):
        tracks[i].cov = cov[i]
    ptcris = [padova_tracks.critical_point.critical_point(p)
              for p in ptcri_names]
    tracks = [ptcris[i].load_eeps(tracks[i], sandro=False)
              for i in range(len(tracks))]
    axs = compare_khd(tracks)
    [ax.set_ylim(0, .550) for ax in axs]
    [ax.set_ylabel('$m/M$', fontsize=18) for ax in axs]
    axs[0].set_xlim(480, 501)
    axs[1].set_xlim(530, 544.5)
    axs[2].set_xlim(577, 586.5)
    plt.savefig('/Users/phil/research/clusters/n419/khd_compare_cov_Z0.004_M2.00.png')


def compare_covs(tracks):

    # scrap...
    columns = ['Xsup', 'Ysup', 'Rstar']
    labels = ['Xs', 'Ys', 'R']
    fig, axs = plt.subplots(nrows=len(columns), sharex=True, figsize=(8, 8))
    colors = ['k', 'b', 'r']
    fmt = '%(cov).1f '
    for i, track in enumerate(tracks):
        print(track.cov, track.mass, track.data.AGE[track.iptcri[5]]/1e6,
              track.data.AGE[track.iptcri[10]]/1e6)
        xdata = track.data.AGE/1e6
        track.calc_core_mu()
        for ax, col, lab in zip(axs, columns, labels):
            if len(col) == 0:
                ydata = track.muc
            else:
                ydata = track.data[col]
            ax.plot(xdata, ydata, lw=3, color=colors[i],
                    label='$\Lambda_c=%.1f$' % track.cov)

            ax.set_ylabel('$%s$' % lab)
            # ax.set_ylim(np.min(ydata), np.max(ydata))
            # ax.yaxis.set_major_locator(MaxNLocator(4))
            # ax.xaxis.set_major_formatter(NullFormatter())

            ax.vlines(track.data.AGE[tracks[i].iptcri[5]]/1e6, *ax.get_ylim(),
                      color='grey', lw=2)
            ax.vlines(track.data.AGE[tracks[i].iptcri[10]]/1e6, *ax.get_ylim(),
                      color='grey', lw=2)

    [ax.set_xlim(475, 600) for ax in axs]


def eep_summary_table(tracks, outfmt='latex', isort='mass', diff_table=False):
    fmt = '%.1f %g %g %.2e %.2e %.2e %.2e %.2e %.2e \n'

    def keyfunc(track):
        return tuple((track.mass, track.Z))

    def rel_diff(ref, x, attr):
        rd = (x.__getattribute__(attr) - ref.__getattribute__(attr)) \
              / ref.__getattribute__(attr)
        return rd

    if outfmt == 'latex':
        fmt = fmt.replace(' ', ' & ').replace('\n', ' \\\\ \n')

    outstr = ['OV Z Mass MSTOage MSTOradius tauH TRGBage TRGBRadius tauHe \n']
    ts = padova_tracks.TrackSet()
    if type(tracks[0]) == str:
        print('loading tracks')
        tracks = [padova_tracks.Track(t) for t in tracks]

    for t in tracks:
        t.cov = float(t.base.split('OV')[1].split('_')[0])

    if isort == 'mass':
        tracks = sorted(tracks, key=keyfunc)

    ts.tracks = tracks
    ts.tracks = ts._load_ptcri(ptcri_loc, sandro=False)
    for t in ts.tracks:
        if t.flag is not None or len(t.iptcri) < 5:
            continue
        t.ageMSTO = t.data.AGE[t.iptcri[5]]
        t.ageTRGB = t.data.AGE[t.iptcri[10]]
        t.rMSTO = t.data.Rstar[t.iptcri[5]]
        t.rTRGB = t.data.Rstar[t.iptcri[10]]
        # could cut at ms_beg instead of LX > 0... this keeps physical argument
        hburn, = np.nonzero((t.data.QH1 == 0) & (t.data.LX > 0))
        heburn, = np.nonzero((t.data.QHE1 == 0) & (t.data.LY > 0))
        t.tH = np.sum(t.data.Dtime[hburn])
        t.tHe = np.sum(t.data.Dtime[heburn])
        outstr.append(fmt % (t.cov, t.Z, t.mass, t.ageMSTO, t.rMSTO, t.tH,
                             t.ageTRGB, t.rTRGB, t.tHe))

    outstr.append('\n\n OV Z Mass MSTOage MSTOradius tauH TRGBage TRGBRadius tauHe \n')
    if diff_table:
        fmt2 = fmt.replace('.2e', '+.2f')
        t05, t05_names = zip(*[(t, t.name) for t in ts.tracks if t.cov == 0.5])
        t05 = list(t05)
        t05_names = list(t05_names)
        for t in ts.tracks:
            if t.cov == 0.5:
                outstr.append(fmt % (t.cov, t.Z, t.mass, t.ageMSTO, t.rMSTO,
                                     t.tH, t.ageTRGB, t.rTRGB, t.tHe))
                continue
            ind = t05_names.index(t.name)
            dageMSTO = rel_diff(t05[ind], t, 'ageMSTO')
            drMSTO = rel_diff(t05[ind], t, 'rMSTO')
            dtH = rel_diff(t05[ind], t, 'tH')
            dageTRGB = rel_diff(t05[ind], t, 'ageTRGB')
            drTRGB = rel_diff(t05[ind], t, 'rTRGB')
            dtHe = rel_diff(t05[ind], t, 'tHe')
            outstr.append(fmt2 % (t.cov, t.Z, t.mass, dageMSTO, drMSTO, dtH,
                                  dageTRGB, drTRGB, dtHe))
    return outstr


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # cov_testgrid_plots()
    match_diagnostic_plots()
