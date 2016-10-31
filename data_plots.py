from __future__ import print_function
import os
import sys
import argparse
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from astropy.io import fits
from bokeh.plotting import ColumnDataSource, figure, gridplot
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.models import Range1d

import seaborn as sns

from match.scripts.utils import parse_pipeline

FIGEXT = '.pdf'


def bplot_cmd_xy(obs, filter1, filter2, xyfile=None):
    """
    2 panel bokeh plot
    """
    color, mag, _, _, good, x, y = load_obs(obs, filter1, filter2,
                                            xyfile=xyfile)

    pid, target = os.path.split(obs)[1].split('_')[:2]
    title = '{} {}'.format(pid, target)
    xlabel = '{}-{}'.format(filter1, filter2)
    ylabel = '{}'.format(filter2)
    outfile = os.path.split(obs)[1] + '.html'

    data = np.column_stack((color[good], mag[good], x[good], y[good]))

    df = pd.DataFrame(data, columns=['color', 'mag2', 'x', 'y'])
    source = ColumnDataSource(df)

    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,save"
    plot_config = dict(plot_width=500, plot_height=500, tools=tools)
    cs = ['#8C3B49', '#212053']

    p1 = figure(title=title, **plot_config)
    p1.circle("color", "mag2", size=1, source=source, color=cs[0])
    p1.y_range = Range1d(26, 14)
    p1.yaxis.axis_label = ylabel
    p1.xaxis.axis_label = xlabel

    p2 = figure(**plot_config)
    p2.circle(data[:, 2], data[:, 3], size=1, source=source, color=cs[1])
    p2.yaxis.axis_label = "Y"
    p2.xaxis.axis_label = "X"

    p = gridplot([[p1, p2]])

    html = file_html(p, CDN, title)

    with open(outfile, 'w') as f:
        f.write(html)
    print('wrote {}'.format(outfile))
    # script, div = components(p)


def replace_all(text, dic):
    """perfrom text.replace(key, value) for all keys and values in dic"""
    for old, new in dic.iteritems():
        text = text.replace(old, new)
    return text


def _plot_cmd(color, mag, color_err=None, mag_err=None, inds=None, ax=None,
              plt_kw=None):
    '''plot a cmd with errors'''
    if inds is None:
        inds = np.arange(len(mag))

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 12))

    plt_kw = plt_kw or {}
    default = {'color': 'black', 'ms': 3}
    default.update(plt_kw)

    ax.plot(color[inds], mag[inds], 'o', **default)

    if color_err is not None and mag_err is not None:
        ax.errorbar(color[inds], mag[inds], fmt='none',
                    xerr=color_err[inds], yerr=mag_err[inds],
                    capsize=0, ecolor='gray')
    return ax


def add_inset(ax0, extent, xlim, ylim):
    '''add an inset axes to the plot and a rectangle on the main plot'''
    ax = plt.axes(extent)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim[::-1])
    rect = plt.Rectangle((xlim[0], ylim[0]), np.diff(xlim), np.diff(ylim),
                         fill=False, color='k')
    ax0.add_patch(rect)
    return ax


def load_obs(filename, filter1, filter2, xyfile=None, fextra='VEGA'):
    if xyfile is not None:
        _, _, x, y = np.loadtxt(xyfile, unpack=True)
    else:
        x = np.array([])
        y = np.array([])

    if filename.endswith('fits'):
        try:
            data = fits.getdata(filename)
            keyfmt = '{}_{}'
            errfmt = '{}_ERR'
            mag = gal.data[keyfmt.format(filter2, fextra)]
            mag1 = gal.data[keyfmt.format(filter1, fextra)]
            color = mag1 - mag
            mag_err = gal.data[errfmt.format(filter2)]
            color_err = \
                np.sqrt(gal.data[errfmt.format(filter1)] ** 2 + mag_err ** 2)
            x = gal.data.X
            y = gal.data.Y
        except ValueError:
            print('Problem with {}'.format(filename))
            return None, None
    elif filename.endswith('match'):
        mag1, mag2 = np.genfromtxt(filename, unpack=True)
        color = mag1 - mag2
        mag = mag1
        mag_err = None
        color_err = None
    elif filename.endswith('dat'):
        try:
            _, x, y, mag, mag_err, color, color_err, _, _ = \
                np.loadtxt(filename, unpack=True)
        except:
            print("Can't understand file format {}".format(filename))
            return None, None
    else:
        _, x, y, mag, mag_err, color, color_err = \
            np.loadtxt(filename, unpack=True)

    good, = np.nonzero((np.abs(color) < 30) & (np.abs(mag) < 30))
    return color, mag, color_err, mag_err, good, x, y


def add_rect(ax, xlim, ylim, kw=None):
    default = {'fill': False, 'color': 'grey', 'lw': 2, 'zorder': 1000}
    kw = kw or {}
    default.update(kw)
    rect = plt.Rectangle((xlim[0], ylim[0]), np.diff(xlim), np.diff(ylim),
                         **default)
    ax.add_patch(rect)
    return

def setup_zoomgrid():
    fig = plt.figure(figsize=(8, 6.5))
    # cmd grid is one square taking up 4 of the 6 axes
    ax = plt.subplot2grid((2,3), (0,0), rowspan=2, colspan=2)
    # for HB probably
    ax2 = plt.subplot2grid((2,3), (0,2))
    # for MSTO probably
    ax3 = plt.subplot2grid((2,3), (1,2))

    plt.subplots_adjust(wspace=0.15, right=0.88)
    ax.tick_params(right=False, top=False)
    for ax_ in [ax2, ax3]:
        ax_.tick_params(labelright=True, labelleft=False,
                        left=False, top=False)
    return fig, (ax, ax2, ax3)


def adjust_zoomgrid(ax, ax2, ax3, zoom1_kw=None, zoom2_kw=None):
    """
    zoom grid is just a fancy call to set_[x,y]lim and plot a
    rectangle.
    """
    default1 = {'xlim': [0.5, 1.],
                'ylim': [19.6, 21.7]}
    default2 = {'xlim': [0.85, 1.1],
                'ylim': [18.2, 19.2]}
    zoom1_kw = zoom1_kw or default1
    zoom2_kw = zoom2_kw or default2
    add_rect(ax, **zoom1_kw)
    add_rect(ax, **zoom2_kw)
    ax2.set_xlim(zoom1_kw['xlim'])
    ax2.set_ylim(np.sort(zoom1_kw['ylim'])[::-1])
    ax3.set_xlim(zoom2_kw['xlim'])
    ax3.set_ylim(np.sort(zoom2_kw['ylim'])[::-1])
    for ax_ in [ax2, ax3]:
        ax_.locator_params(axis='x', nbins=4)
        ax_.locator_params(axis='y', nbins=6)
    return [ax, ax2, ax3]


def cmd_axeslimits(filter1, xlim=None, ylim=None):
    """
    default cmd limits for IR or optical
    """
    if xlim is None:
        if filter1 == "F160W" or filter1 == "F110W":
            xlim = [-5, 2]
        else:
            xlim = [-0.5, 4]
    if ylim is None:
        if filter1 == "F160W" or filter1 == "F110W":
            ylim = [28., 14]
        else:
            ylim = [26., 14]
    return xlim, ylim


def cmd(obs, filter1, filter2, zoom=False, scatter=False, xlim=None, ylim=None,
        xy=True, fig=None, axs=None, plt_kw=None, zoom1_kw=None, zoom2_kw=None):
    '''
    plot cmd of data, two insets are hard coded.
    '''
    plt_kw = plt_kw or {}
    color, mag, color_err, mag_err, good, x, y = \
        load_obs(obs, filter1, filter2)

    if axs is None:
        if not xy:
            if zoom:
                fig, (ax, ax2, ax3) = setup_zoomgrid()
            else:
                fig, ax = plt.subplots(figsize=(12, 12))
        else:
            fig, (ax, axxy) = plt.subplots(ncols=2, figsize=(16, 8))
    else:
        if xy:
            ax, axxy = axs
        elif zoom:
            ax, ax2, ax3 = axs
        else:
            ax = axs

    ax = _plot_cmd(color, mag, color_err=color_err, mag_err=mag_err, inds=good,
                   ax=ax, plt_kw=plt_kw)

    ax.set_ylabel(r'${}$'.format(filter1))
    ax.set_xlabel(r'${}-{}$'.format(filter1, filter2))
    xlim, ylim = cmd_axeslimits(filter1, xlim=xlim, ylim=ylim)
    ax.set_xlim(xlim)
    ax.set_ylim(np.sort(ylim)[::-1])
    axs = [ax]

    if zoom:
        ax2 = _plot_cmd(color, mag, color_err=color_err, mag_err=mag_err,
                        inds=good, ax=ax2, plt_kw=plt_kw)
        ax3 = _plot_cmd(color, mag, color_err=color_err, mag_err=mag_err,
                        inds=good, ax=ax3, plt_kw=plt_kw)
        axs = adjust_zoomgrid(ax, ax2, ax3, zoom1_kw=zoom1_kw, zoom2_kw=zoom2_kw)

    if xy:
        plt_kw = plt_kw or {}
        default = {'color': 'k', 'ms': 3}
        default.update(plt_kw)
        axxy.scatter(x[good], y[good], s=star_size(mag[good]))
        axxy.set_ylabel(r'$\alpha \rm{(deg)}$')
        axxy.set_xlabel(r'$\delta \rm{(deg)}$')
        axs.append(axxy)

    return fig, axs


def star_size(mag_data):

    '''
    Convert magnitudes into intensities and define sizes of stars in
    finding chart.
    '''
    # Scale factor.
    factor = 500. * (1 - 1 / (1 + 150 / len(mag_data) ** 0.85))
    return 0.1 + factor * 10 ** ((np.array(mag_data) - min(mag_data)) / -2.5)

# def overplot_iso(data):
#     data = np.genfromtxt('/Users/phil/Downloads/output113116546142.dat')
#     fig, axs = data_plots.cmd('../10396_NGC419-HRC.gst.fits', 'F555W_VEGA',
#                               'F814W_VEGA', True)
#     mag2 = rsp.astronomy_utils.Mag2mag(data.T[21], 'F814W', 'acs_hrc',
#                                         dmod=dmod, Av=av)
#     mag1 = rsp.astronomy_utils.Mag2mag(data.T[15], 'F555W', 'acs_hrc',
#                                          dmod=dmod, Av=av)
#     icolor = mag1-mag2
#     [ax.plot(icolor, mag2, '.', alpha=0.5, color='blue') for ax in axs]


def unique_inds(arr):
    '''return unique values and array of indicies matching the unique value'''
    un_arr = np.unique(arr)
    iarr = np.digitize(arr, bins=un_arr) - 1
    return un_arr, iarr


def plot_isochrone_grid(iso_files, ax_by='age'):
    isos = [fileio.readfile(i, col_key_line=1) for i in iso_files]
    fnames = [i.split('/')[-1].replace('.dat', '') for i in iso_files]

    ovs = np.array([i.split('_')[2].replace('OV', '') for i in fnames],
                   dtype=float)
    un_ovs, iovs = unique_inds(ovs)
    ovstr = r'$\Lambda_c=%.1f$'

    ages = np.array([i.split('_')[3].replace('age', '') for i in fnames],
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

    colors = brewer2mpl.get_map('RdYlBu', 'Diverging', 5).mpl_colors
    # colors = rsp.graphics.discrete_colors(len(labs))
    # colors = ['red', 'black', 'blue', 'orange', 'green']

    # plot the isochrones, each panel at one age, colored by cov
    for i, iso in enumerate(isos):
        axs[iax[i]].plot(iso['logTe'], iso['logLLo'], color=colors[icols[i]],
                         alpha=0.5)

    # fake the legend
    [axs[ileg].plot(-99, -99, color=colors[i], lw=3, alpha=0.3,
                    label=labfmt % labs[i]) for i in range(len(labs))]
    axs[ileg].legend(loc=2)

    for i, ax in enumerate(axs):
        ax.set_xlabel(r'$\log T_{\rm eff}\ (K)$')
        ax.grid(color='k')
        ax.annotate(annfmt % anns[i], (3.80, 0.7))
    axs[0].set_ylim(0.6, 2.2)
    axs[0].set_xlim(4.02, 3.65)
    axs[0].set_ylabel(r'$\log L\ (L_\odot)$')


def get_filters(fitsfile):
    h = fits.getheader(fitsfile)
    return [h for h in h['filters'].split(',') if len(h) > 0]


def main(argv):
    parser = argparse.ArgumentParser(description="Plot CMD")

    parser.add_argument('--outfile', type=str, default=None,
                        help='output image to write to ([obs].png)')

    parser.add_argument('-b', '--bokeh', action='store_true',
                        help='make bokeh tables')

    parser.add_argument('-p', '--png', action='store_true',
                        help='make FIGEXT')

    parser.add_argument('-y', '--yfilter', type=str, default='I',
                        help='V or I to plot on yaxis (I)')

    parser.add_argument('-x', '--xyfile', type=str, default=None,
                        help='xyfile to plot (read in obs)')

    parser.add_argument('-f', '--filters', type=str, default=None,
                        help='comma separated filter list. (from filename)')

    parser.add_argument('-s', '--scatter', action='store_true',
                        help='make a scatter contour plot')

    parser.add_argument('--clobber', action='store_true',
                        help='overwrite outfile if exists')

    parser.add_argument('--pdb', action='store_true',
                        help='invoke pdb')

    parser.add_argument('obs', type=str, nargs='*',
                        help='data file to make CMD')

    args = parser.parse_args(argv)
    if args.pdb:
        import pdb
        pdb.set_trace()

    for obs in args.obs:
        if args.filters is not None:
            filters = args.filters.split(',')
        else:
            _, filters = parse_pipeline(obs)
        if len(filters) == 1:
            if obs.endswith('.fits'):
                filters = get_filters(obs)
            print('Error only one filter {}.'.format(obs))
            print('perhaps see .pipeline_filenames.main()')
            continue

        # Reddest should be filter2
        if 'F814W' in filters:
            idx = filters.index('F814W')
        if 'F110W' in filters:
            idx = filters.index('F110W')
        if 'F160W' in filters:
            idx = filters.index('F160W')
        filter2 = filters.pop(idx)

        for filter1 in filters:
            # either a supplied outputfile, the obs name + .png
            # or if there are more than 2 filters, the two plotted .png
            outfile = args.outfile
            if outfile is None:
                outfile = obs + FIGEXT
                if len(filters) > 1:
                    # take out the filters not being plotted
                    try:
                        notfs = [f for f in filters if filter1 not in f]
                        outfile = replace_all(obs + '.png',
                                              dict(zip(notfs,
                                                       ['']*len(notfs))))
                        # ^ leaves _-F336W-F814W or F110W----F814W so:
                        uch = {'--': '-', '_-': '_', '-_': '_'}
                        outfile = replace_all(
                            replace_all(replace_all(outfile, uch), uch), uch)
                    except ValueError:
                        raise
                        return

            if args.bokeh:
                bplot_cmd_xy(obs, filter1, filter2)

            if args.png:
                if os.path.isfile(outfile) and not args.clobber:
                    print('not overwriting {}'.format(outfile))
                    continue

                _, axs = cmd(obs, filter1, filter2, inset=False,
                             scatter=args.scatter, xyfile=args.xyfile)
                if axs is not None:
                    plt.savefig(outfile)
                    print('wrote {}'.format(outfile))
                    plt.close()

if __name__ == '__main__':
    sns.set_context('paper', font_scale=2)
    sns.set_style('whitegrid')
    sd = sns.axes_style()
    sd['text.usetex'] = True
    sns.set(sd)
    main(sys.argv[1:])
