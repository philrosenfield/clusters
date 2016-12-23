import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from astropy.io import fits
from shapely.geometry import Polygon
from matplotlib.path import Path

from .footprints.footprint_overlaps import parse_poly

sns.set()


def cullfits(fitsfile, astecafile):
    data = fits.getdata(fitsfile)
    ra, dec = np.loadtxt(astecafile, usecols=(1, 2), unpack=True)

    ply = Polygon(np.column_stack((ra, dec)))
    preg = ply.convex_hull
    verts = parse_poly(preg.wkt)
    mask = Path(verts).contains_points(np.column_stack((data.RA, data.DEC)))

    f1, f2 = [a for a in data.columns.names if a.endswith('VEGA')]
    if 'F814W' not in f2:
        f2, f1 = f1, f2

    outfile = astecafile.replace('gst_memb.dat', 'gst_rcl.match')
    np.savetxt(outfile, np.column_stack(
        (data[f1][mask], data[f2][mask])), fmt='%.4f')
    print('wrote {}'.format(outfile))
    return


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Cull fits file to asteca derived star cluster radius")

    parser.add_argument('fitsfile', type=str, help='photometry fits file')

    parser.add_argument('astecafile', type=str, help='asteca input file')

    return args


def main(argv):
    args = parse_args(argv)
    cullfits(args.fitsfile, args.astecafile)


if __name__ == "__main__":
    sys.exit(main())
