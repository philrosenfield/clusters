from __future__ import print_function
import os
import sys
import argparse
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from astropy.io import fits
from bokeh.plotting import ColumnDataSource, figure, gridplot
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.models import Range1d

from .utils import replace_all
from match.scripts.utils import parse_pipeline

FIGEXT = '.png'
plt.style.use('presentation')

sns.set_context('paper', font_scale=2)
sns.set_style('ticks')
sns.set_style('whitegrid')
sd = sns.axes_style()
sd['text.usetex'] = True
sns.set(sd)

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


def _plot_cmd(color, mag, color_err=None, mag_err=None, inds=None, ax=None,
              plt_kw=None):
    '''plot a cmd with errors'''
    if inds is None:
        inds = np.arange(len(mag))

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 12))

    plt_kw = plt_kw or {}
    default = {'color': 'black', 'ms': 3, 'rasterized': True}
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


def load_obs(filename, filter1, filter2, xyfile=None, fextra='VEGA',
             crowd=None):
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
            if crowd is not None:
                crd = keyfmt.format(filter2, 'CROWD')
                crd1 = keyfmt.format(filter1, 'CROWD')
                inds, = np.nonzero((data[crd] < crowd) & (data[crd1] < crowd))
                data = data[inds]
            mag2 = data[keyfmt.format(filter2, fextra)]
            mag = data[keyfmt.format(filter1, fextra)]
            color = mag - mag2
            mag_err = data[errfmt.format(filter1)]
            color_err = \
                np.sqrt(data[errfmt.format(filter1)] ** 2 + mag_err ** 2)
            x = data.X
            y = data.Y
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
    """
    Set up a 6 panel plot, 2x2 grid is one main axes and the other 2 are
    zoom-ins of the main axes
    """
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


def adjust_zoomgrid(ax, ax2, ax3, zoom1_kw=None, zoom2_kw=None, reversey=True):
    """
    zoom grid is just a fancy call to set_[x,y]lim and plot a
    rectangle.
    """
    def adjust(ax, axz, zoom, reversey=True):
        add_rect(ax, **zoom)
        axz.set_xlim(zoom['xlim'])
        ylim = np.sort(zoom['ylim'])
        if reversey:
            ylim = np.sort(zoom['ylim'])[::-1]
        axz.set_ylim(ylim)
        return ax

    default1 = {'xlim': [0.5, 1.],
                'ylim': [19.6, 21.7]}
    zoom1_kw = zoom1_kw or default1

    adjust(ax, ax2, zoom1_kw, reversey=reversey)
    if zoom2_kw is not None:
        adjust(ax, ax3, zoom2_kw, reversey=reversey)

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
        xy=True, fig=None, axs=None, plt_kw=None, zoom1_kw=None, zoom2_kw=None,
        load_obskw=None):
    '''
    plot cmd of data, two insets are hard coded.
    '''
    plt_kw = plt_kw or {}
    load_obskw = load_obskw or {}
    color, mag, color_err, mag_err, good, x, y = \
        load_obs(obs, filter1, filter2, **load_obskw)

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
                h = fits.getheader(obs)
                filters = [h for h in h['filters'].split(',') if len(h) > 0]
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
