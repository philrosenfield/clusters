from astropy.io import fits
import numpy as np
import seaborn; seaborn.set()
import matplotlib.pyplot as plt
from match.scripts import graphics
import os

def stellar_prob(obs, mod, normalize=False):
    '''
    FROM MATCH README
    The quality of the fit is calculated using a Poisson maximum likelihood
    statistic, based on the Poisson equivalent of chi^2.
      2 m                                if (n=0)
      2 [ 0.001 + n * ln(n/0.001) - n ]  if (m<0.001)
      2 [ m + n * ln(n/m) - n ]          otherwise
    m=number of model points; n=number of observed points

    This statistic is based on the Poisson probability function:
       P =  (e ** -m) (m ** n) / (n!),
    Recalling that chi^2 is defined as -2lnP for a Gaussian distribution and
    equals zero where m=n, we treat the Poisson probability in the same
    manner to get the above formula.

    '''
    n = obs
    m = mod

    if normalize is True:
        n /= np.sum(n)
        m /= np.sum(m)

    d = 2. * (m + n * np.log(n / m) - n)
    smalln = np.abs(n) < 1e-10
    d[smalln] = 2. * m[smalln]

    smallm = (m < 0.001) & (n != 0)
    d[smallm] = 2. * (0.001 + n[smallm] * np.log(n[smallm] / 0.001) - n[smallm])

    sig = np.sqrt(d) * np.sign(n - m)

    prob = np.exp( -1 * np.sum(d) / 2)

    return d, prob, sig

def make_hess(cmd, mbinsize, cbinsize=None, cbin=None, mbin=None):
    """
    Compute a hess diagram (surface-density CMD) on photometry data.

    Paramters
    ---------
    color: ndarray
        color values

    mag: ndarray
        magnitude values

    binsize: sequence
        width of bins, in magnitudes

    cbin: sequence, optional
        set the centers of the color bins

    mbin: sequence, optional
        set the centers of the magnitude bins

    cbinsize: sequence, optional
        width of bins, in magnitudes

    Returns
    -------
    Cbin: sequence
        the centers of the color bins

    Mbin: sequence
        the centers of the magnitude bins

    Hess:
        The Hess diagram array

    """
    if mbin is None:
        mbin = np.arange(cmd[1].min(), cmd[1].max(), mbinsize)
    else:
        mbin = np.array(mbin).copy()

    if cbin is None:
        if cbinsize is None:
            cbinsize = mbinsize
        cbin = np.arange(cmd[0].min(), cmd[0].max(), cbinsize)
    else:
        cbin = np.array(cbin).copy()

    hess, cedges, medges = np.histogram2d(cmd[0], cmd[1], bins=[cbin, mbin])
    return hess, cedges, medges, cbin, mbin



def same_limits(cmd1, cmd2, cull=False):
    def good(cmd, thresh=50.):
        return np.nonzero((np.abs(cmd[0]) < 50.) & (np.abs(cmd[1]) < 50.))[0]

    def get_extent(cmd):
        return [np.min(cmd[0]), np.max(cmd[0]), np.min(cmd[1]), np.max(cmd[1])]

    def extrema(arr):
        return arr[np.argmax([np.abs(arr)])]

    def largest_shape(cmd1, cmd2):
        ext1 = get_extent(cmd1)
        ext2 = get_extent(cmd2)
        return [extrema([e1, e2]) for e1, e2 in zip(ext1, ext2)]

    def within(cmd, extent):
        from matplotlib.path import Path
        x1, x2, y1, y2 = extent
        verts = [[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]
        idx, = np.nonzero(Path(verts).contains_points(cmd.T))
        return idx

    # Recovered stars
    idx1 = good(cmd1)
    idx2 = good(cmd2)

    extent = largest_shape(cmd1[:, idx1], cmd2[:, idx2])

    retv1 = within(cmd1, extent)
    retv2 = within(cmd2, extent)

    if cull:
        retv1 = cmd1[:, retv1]
        retv2 = cmd2[:, retv2]

    return retv1, retv2, extent


def compare_data(v1, v2):
    target = v1.split('_')[1].split('.')[0]

    v1data = fits.getdata(v1)
    v2data = fits.getdata(v2)

    filters = [c for c in v1data.dtype.names if c.endswith('VEGA')]
    filtcombos = [[f, filters[-1]] for f in filters[:-1]]

    for filtcombo in filtcombos:
        filter1 = filtcombo[0]
        filter2 = filtcombo[1]

        strfilt1, strfilt2 = [c.replace('_VEGA', '') for c in [filter1, filter2]]

        yfilter = filter2
        stryfilt = strfilt2

        color1 = v1data[filter1] - v1data[filter2]
        color2 = v2data[filter1] - v2data[filter2]


        magv1 = v1data[yfilter]
        magv2 = v2data[yfilter]

        cmd1 = np.vstack([color1, magv1])
        cmd2 = np.vstack([color2, magv2])

        cmd1, cmd2, ext = same_limits(cmd1, cmd2, cull=True)

        hess1, cedges, medges, cbin, mbin = make_hess(cmd1, 0.05)
        hess2 = make_hess(cmd2, 0.05, mbin=mbin, cbin=cbin)[0]

        dif = hess1 - hess2
        sig = stellar_prob(hess1, hess2)[-1]
        hesses = [hess1.T, hess2.T, dif.T, sig.T]

        grid = graphics.match_plot(hesses, ext, labels=['{} v1'.format(target), '{} v2'.format(target), 'Diff', 'Sig'])
        [ax.set_xlabel('{}-{}'.format(strfilt1, strfilt2)) for ax in grid.axes_row[1]]
        [ax.set_ylabel('{}'.format(stryfilt)) for ax in grid.axes_column[0]]
        grid.axes_all[0].xaxis.label.set_visible(True)

        outfile = '{}_{}_{}.png'.format(target, strfilt1, strfilt2)
        plt.savefig(outfile, dpi=300)
        print('wrote {}'.format(outfile))
    return

if __name__ == "__main__":
    fnames = [l for l in os.listdir('.') if l.endswith('fits')]
    v1s = [v for v in fnames if 'v1' in v]
    v2s = [v for v in fnames if not 'v1' in v]
    for v1, v2 in zip(v1s, v2s):
        assert (v1.split('.')[0] == v2.split('.')[0]), 'files do not match {} {}'.format(v1, v2)
        compare_data(v1, v2)

