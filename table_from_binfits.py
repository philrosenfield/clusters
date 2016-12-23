import glob
import os
import sys
import argparse

import pandas as pd
import numpy as np

from astropy.io import fits
from shapely.geometry import Polygon
from shapely import geos

wkt = geos.WKTWriter(geos.lgeos)
wkt.rounding_precision = 8

def convex_hull_(xarr, yarr):
    """call shapely to calculate convex_hull on scatter points."""
    ply = Polygon(np.column_stack([xarr, yarr]))
    plyc = ply.convex_hull
    return wkt.write(plyc), plyc.centroid.x, plyc.centroid.y

def build_table(fnames, outcsv='table.csv', xycoords=False):
    """build a table using binary fits format data"""
    xcoord = 'ra'
    ycoord = 'dec'
    if xycoords:
        xcoord = 'x'
        ycoord = 'y'
    columns = ['pid', 'target', 'filters', xcoord, ycoord, 'images', 's_region']
    table = pd.DataFrame(columns=columns)
    for fname in fnames:
        df = pd.DataFrame(columns=columns)
        pid, target, filters, _ = fname.split('_')
        data = fits.getdata(fname)
        s_region, xcenter, ycenter = convex_hull_(data[xcoord], data[ycoord])
        df['images'] = \
            pd.Series(','.join(glob.glob1(os.getcwd(),
                                          '_'.join([pid, target, '*png']))))
        df[xcoord] = pd.Series(xcenter)
        df[ycoord] = pd.Series(ycenter)
        df['s_region'] = pd.Series(s_region)
        df['pid'] = pd.Series(pid)
        df['target'] = pd.Series(target)
        df['filters'] = pd.Series(filters.split(','))
        table = table.append(df, ignore_index=True)

    table.to_csv(outcsv, index=False, sep=';')

def main(argv):
    parser = argparse.ArgumentParser(description=("Make a table for aladin ",
                                                  "using binary fits tables"))

    parser.add_argument('-o', '--outcsv', type=str, default='table.csv',
                        help='name of output file')

    parser.add_argument('fnames', type=str, nargs='*',
                        help='name(s) fits files for the table')

    args = parser.parse_args(argv)

    build_table(args.fnames, outcsv=args.outcsv)

if __name__ == "__main__":
    main(sys.argv[1:])
