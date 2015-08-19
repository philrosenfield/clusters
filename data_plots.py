import palettable
import pyfits
import ResolvedStellarPops as rsp
import matplotlib
import matplotlib.pylab as plt
import numpy as np

def _plot_cmd(color, mag, color_err=None, mag_err=None, inds=None, ax=None):
    '''plot a cmd with errors'''
    if inds is None:
        inds = np.arange(len(mag))

    if ax is None:
        fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(color[inds], mag[inds], '.', color='black', ms=4)
    if not None in [color_err, mag_err]:
        ax.errorbar(color[inds], mag[inds], fmt=None, xerr=color_err[inds],
                    yerr=mag_err[inds], capsize=0, ecolor='gray')
    return ax


def add_inset(ax0, extent, xlims, ylims):
    '''add an inset axes to the plot and a rectangle on the main plot'''
    ax = plt.axes(extent)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims[::-1])
    rect = matplotlib.patches.Rectangle((xlims[0], ylims[0]),
                                        np.diff(xlims), np.diff(ylims),
                                        fill=False, color='k')
    ax0.add_patch(rect)
    return ax

#fig, axs = cmd('../10396_NGC419-HRC.gst.fits', 'F555W_VEGA', 'F814W_VEGA', True)

def cmd(fitsfile, filter1, filter2, inset=False):
    '''
    plot cmd of data, two insets are hard coded.
    '''
    gal = rsp.StarPop()
    gal.data = pyfits.getdata(fitsfile)
    mag = gal.data[filter2]# - 0.35
    mag1 = gal.data[filter1]# - 0.1
    color = mag1 - mag

    filt1 = filter1.split('_')[0]
    filt2 = filter2.split('_')[0]
    fextra = '(%s)' % filter2.split('_')[1]
    mag_err = gal.data['%s_ERR' % filt2]
    color_err = np.sqrt(gal.data['%s_ERR' % filt1] ** 2 + mag_err ** 2)
    good, = np.nonzero((np.abs(color)<30) & (np.abs(mag) < 30))
    fig, ax = plt.subplots(figsize=(12, 12))
    ax = _plot_cmd(color, mag, color_err=color_err, mag_err=mag_err, inds=good, ax=ax)

    plt.tick_params(labelsize=18)
    #ax.set_ylabel(r'$%s\ %s$' % (filt2, fextra), fontsize=24)
    #ax.set_xlabel(r'$%s-%s\ %s$' % (filt1, filt2, fextra), fontsize=24)
    ax.set_ylabel(r'$%s$' % filt2, fontsize=24)
    ax.set_xlabel(r'$%s-%s$' % (filt1, filt2), fontsize=24)
    ax.set_ylim(26., 14)
    ax.set_xlim(-0.5, 4)
    if inset:
        ax1 = add_inset(ax, [0.45, 0.45, 0.42, 0.3], [0., 0.6], [19.6, 21.7])
        ax1 = _plot_cmd(color, mag, color_err=color_err, mag_err=mag_err,
                        inds=good, ax=ax1)
        ax2 = add_inset(ax, [0.18, 0.74, .19, .15], [0.85, 1.1], [18.2, 19.2])
        ax2 = _plot_cmd(color, mag, color_err=color_err, mag_err=mag_err,
                        inds=good, ax=ax2)
        return fig, (ax, ax1, ax2)
    else:
        return fig, ax


def overplot_iso(data):
    data = np.genfromtxt('/Users/phil/Downloads/output113116546142.dat')
    fig, axs = data_plots.cmd('../10396_NGC419-HRC.gst.fits', 'F555W_VEGA', 'F814W_VEGA', True)
    mag2 = rsp.astronomy_utils.Mag2mag(data.T[21], 'F814W', 'acs_hrc', dmod=dmod, Av=av)
    mag1 = rsp.astronomy_utils.Mag2mag(data.T[15], 'F555W', 'acs_hrc', dmod=dmod, Av=av)
    icolor = mag1-mag2
    [ax.plot(icolor, mag2, '.', alpha=0.5, color='blue') for ax in axs]


def unique_inds(arr):
    '''return unique values and array of indicies matching the unique value'''
    un_arr = np.unique(arr)
    iarr = np.digitize(arr, bins=un_arr) - 1
    return un_arr, iarr

def plot_isochrone_grid(iso_files, ax_by='age'):
    isos = [rsp.fileio.readfile(i, col_key_line=1) for i in iso_files]
    fnames = [i.split('/')[-1].replace('.dat', '') for i in iso_files]

    ovs = np.array([i.split('_')[2].replace('OV','') for i in fnames],
                   dtype=float)
    un_ovs, iovs = unique_inds(ovs)
    ovstr = r'$\Lambda_c=%.1f$'

    ages = np.array([i.split('_')[3].replace('age','') for i in fnames],
                     dtype=float)
    un_ages, iages = unique_inds(ages)
    agestr = r'$\log Age=%.1f$'

    if ax_by == 'age':
        nax = len(un_ages)
        iax = iages
        icols = iovs
        labs = un_ovs
        labfmt = ovstr
        anns = un_ages
        annfmt = agestr
        ileg = -1
    else:
        nax = len(un_ovs)
        iax = iovs
        icols = iages
        labs = un_ages
        labfmt = agestr
        anns = un_ovs
        annfmt = ovstr
        ileg = 0

    fig, axs = plt.subplots(ncols=nax, figsize=(20, 6), sharex=True,
                            sharey=True)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, wspace=0.08)

    #colors = palettable.get_map('RdYlBu', 'Diverging', 5).mpl_colors
    colors = rsp.graphics.discrete_colors(len(labs))
    #colors = ['red', 'black', 'blue', 'orange', 'green']

    # plot the isochrones, each panel at one age, colored by cov
    for i, iso in enumerate(isos):
        axs[iax[i]].plot(iso['logTe'], iso['logLLo'], color=colors[icols[i]],
                           alpha=0.5)

    # fake the legend
    [axs[ileg].plot(-99, -99, color=colors[i], lw=3, alpha=0.3,
                  label=labfmt % labs[i]) for i in range(len(labs))]
    axs[ileg].legend(loc=2, fontsize=16)

    for i, ax in enumerate(axs):
        ax.set_xlabel(r'$\log T_{\rm eff}\ (K)$', fontsize=20)
        ax.grid(color='k')
        ax.annotate(annfmt % anns[i], (3.80, 0.7), fontsize=20)
    axs[0].set_ylim(0.6, 2.2)
    axs[0].set_xlim(4.02, 3.65)
    axs[0].set_ylabel(r'$\log L\ (L_\odot)$', fontsize=20)
