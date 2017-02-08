from __future__ import print_function
import argparse
import os
import sys
import seaborn

# from ..utils import replace_all
import numpy as np
import pandas as pd
from collections import OrderedDict
from shapely.geometry import Polygon, Point
from fitshelper.footprints import merge_polygons, parse_poly, group_polygons
# seaborn.set()

def cross_match(lit_cat, mast_cat, plot=False, ra='RAJ2000',
                dec='DEJ2000'):
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

    # mast = pd.read_csv(mast_cat, header=4)
    mast = pd.read_csv(mast_cat,  header=0, skiprows=[0])
    mast['group'] = np.nan
    mast['litidx'] = np.nan

    lit = pd.read_csv(lit_cat,  header=0, skiprows=[1])
    lit['group'] = np.nan

    # column labels may need be be generalized.
    radecs = [Point(r,d) for r,d in zip(lit[ra], lit[dec])]

    plys = [Polygon(parse_poly(mast['s_region'].iloc[i]))
            for i in range(len(mast['s_region']))]

    # Group observations by overlapping footprints
    grouped_plys = group_polygons(plys)

    # For the following search, make one polygon per group
    group = OrderedDict()
    for i, g in enumerate(grouped_plys):
        p, inds = merge_polygons(g)
        group[i] = ({'p': p, 'inds': inds})
        # Assign group id to mast table
        mast['group'].iloc[inds] = i

    # loop through all combinations
    for g, pdict in group.items():
        for i, radec in enumerate(radecs):
            # Ignore single filter observations (1-item groups)
            if len(pdict['inds']) == 1 or \
                len(np.unique(mast.iloc[pdict['inds']].filters)) < 2:
                continue
            # Is the radec point in the s_region?
            if pdict['p'].contains(radec):
                if np.isfinite(lit['group'].iloc[i]):
                    # there are complex regions that are not grouped together
                    # so more than one radec point can be in a polygon.
                    # there has to be a better way than making a single item
                    # string...
                    tmp = lit['group'].iloc[i]
                    lit['group'].iloc[i] = ','.join(['{:g}'.format(t)
                                                     for t in [tmp, g]])
                # add group id to literature table
                lit['group'].iloc[i] = g
                # add literature index to the mast table
                mast['litidx'].iloc[pdict['inds']] = i

                if plot:
                    p = np.array(radec.to_wkt()
                                      .replace('POINT (', '')
                                      .replace(')','').split(), dtype=float)
                    v = parse_poly(pdict['p'].to_wkt())
                    ax.plot(v[:, 0], v[:, 1])
                    ax.plot(p[0], p[1], 'o')

    fins, = np.nonzero(np.isfinite(mast['litidx']))
    print('{} lit values, {} MAST values, {} matches.'.format(len(radecs),
                                                              len(plys),
                                                              len(fins)))
    df = mast.loc[fins]
    df['SimbadName'] = lit.iloc[df['litidx']]['SimbadName'].tolist()
    fname = \
        '{}_matched_{}'.format(os.path.split(mast_cat.replace('.csv',''))[1],
                               os.path.split(lit_cat)[1])
    df.to_csv(fname, index=False)
    #TO DO write out lit.to_csv

    if plot:
        plt.savefig('cross_match.png')
        plt.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="cross match a literature catalog with a MAST catalog")

    parser.add_argument('-p', '--plot', action='store_true',
                        help='make a plot')

    parser.add_argument('--ra', type=str, default='RAJ2000',
                        help='literature ra column name')

    parser.add_argument('--dec', type=str, default='DEJ2000',
                        help='literature dec column name')

    parser.add_argument('lit_cat', type=str,
                        help='literature catalog')

    parser.add_argument('mast_cat', type=str,
                        help='MAST catalog')

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cross_match(args.lit_cat, args.mast_cat, plot=args.plot, ra=args.ra,
                dec=args.dec)

if __name__ == "__main__":
    sys.exit(main())
