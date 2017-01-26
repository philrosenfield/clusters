import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from astropy.io import fits
from shapely.geometry import Polygon
from matplotlib.path import Path

from .footprints.footprint_overlaps import parse_poly

sns.set()


def cullfits(fitsfile, astecafile, formatch=True, diagplot=False):
    print(fitsfile)
    print(astecafile)
    ext = 'VEGA'
    data = fits.getdata(fitsfile)
    ra, dec = np.loadtxt(astecafile, usecols=(1, 2), unpack=True)

    ply = Polygon(np.column_stack((ra, dec)))
    preg = ply.convex_hull
    verts = parse_poly(preg.wkt)
    mask = Path(verts).contains_points(np.column_stack((data.RA, data.DEC)))

    try:
        f1, f2 = [a for a in data.columns.names if a.endswith(ext)]
        if 'F814W' not in f2:
            f2, f1 = f1, f2
    except ValueError:
        f1, f2, f3 = [a for a in data.columns.names if a.endswith(ext)]
        if formatch:
            header = ''
            if 'F814W' not in f2:
                f1 = f2
                f2 = f3
            print(f1, f2)
            dat = np.column_stack([data[f1][mask], data[f2][mask]])
            outfile = astecafile.replace('gst_memb.dat', 'gst_rcl.match')
        else:
            # Hack to work for 3 filter data only ... might be useful for 2
            err = 'ERR'
            f1e = f1.replace('VEGA', err)
            f2e = f2.replace('VEGA', err)
            f3e = f3.replace('VEGA', err)
            dat = np.column_stack([data[f1][mask],data[f1e][mask],
                                   data[f2][mask],data[f2e][mask],
                                   data[f3][mask],data[f3e][mask]])
            header = ' '.join([f1, f1e, f2, f2e, f3, f3e])
    outfile = astecafile.replace('gst_memb.dat', 'gst_rcl.dat')
    np.savetxt(outfile, dat, fmt='%.4f', header=header)
    if diagplot:
        fig, ax = plt.subplots()
        ax.plot(data.RA, data.DEC, '.')
        ax.plot(data.RA[mask], data.DEC[mask], '.')
        plt.savefig(outfile + '.png')
    print('wrote {}'.format(outfile))
    return


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Cull fits file to asteca derived star cluster radius")

    parser.add_argument('fitsfile', type=str, help='photometry fits file')

    parser.add_argument('astecafile', type=str, help='asteca input file')

    parser.add_argument('--all', action="store_false",
                        help='keep all filters, (not for match)')

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    cullfits(args.fitsfile, args.astecafile, formatch=args.all)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
