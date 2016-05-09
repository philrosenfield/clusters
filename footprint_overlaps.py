import pandas as pd
from shapely.geometry import Polygon
import numpy as np
import itertools
import os
import sys
import seaborn; seaborn.set()

def parse_footprint(fname):
    """parse a ds9 linear footprint into a ; separated line: filename, polygon, central ra dec"""
    fmt = '{};{};{}'
    with open(fname) as inp:
        lines = inp.readlines()
    filename = fname.replace('_footprint_ds9_linear.reg', '.fits')
    polygons = [p.strip().split('#')[0] for p in lines if 'polygon' in p]
    points = [p.split('#')[0].strip() for p in lines if 'point' in p]
    texts = [p.replace('text', 'point').split('#')[0].strip()
             for p in lines if 'text' in p]
    coords = points
    if len(polygons) != len(coords):
        if len(polygons) != len(texts):
            # rolling our own...
            pcoords = [parse_poly(p) for p in polygons]
            polys = np.array([Polygon(c) for c in pcoords])
            pfmt = 'point {:.6f} {:.6f}'
            coords = [pfmt.format(p.centroid.x, p.centroid.y) for p in polys]
        else:
            coords = texts

    assert len(coords)==len(polygons), 'mismatch'

    polystr = [fmt.format(fname,polygons[i], coords[i]) for i in range(len(coords))]
    return polystr

def parse_footprints(fnames):
    """create all_poly.csv made from the ds9 footprints of several files"""
    line = '\n'.join('\n'.join([parse_footprint(f) for f in fnames]))
    with open('all_poly.csv', 'w') as outp:
        outp.write(line)


def simplify_footprint(fname):
    """
    parse footprint using parse_footprint and calculate polygons convex
    hull in the case of more than one polygon per file.
    """
    lines = parse_footprint(fname)
    radecs = np.array([np.array(l.strip().split('point')[1].split(), dtype=float) for l in lines])
    s_region = np.array([l.strip().split(';')[1].strip() for l in lines])
    pids, targets = zip(*[l.split(';')[0].split('/')[0].split('_') for l in lines])

    if len(s_region) > 1:
        coords = [parse_poly(p) for p in s_region]
        polys = np.array([Polygon(c) for c in coords])
        polys = [p for p in polys if p.is_valid is True]
        p = polys[0]
        for i in range(len(polys)-1):
            p = p.union(polys[i+1])
        p2 = p.convex_hull
        ns_region = str(p2.boundary).replace('LINESTRING ', 'polygon')
        nradec = 'point {:.6f} {:.6f}'.format(p2.centroid.x, p2.centroid.y)
        return ';'.join([lines[0].split(';')[0], ns_region, nradec])
    if len(lines) == 1:
        return lines[0]


def simplify_footprints(fnames):
    """create all_poly_simp.csv made from the ds9 footprints of several files"""
    line = '\n'.join([simplify_footprint(f) for f in fnames])
    with open('all_poly_simp.csv', 'w') as outp:
        outp.write(line)


def replace_all(text, dic):
    """perfrom text.replace(key, value) for all keys and values in dic"""
    for old, new in dic.iteritems():
        text = text.replace(old, new)
    return text

def read_footprints(filename, instrument=None):
    """
    read output from simplify_footprints or parse_footprints into a DataFrame
    with filename, ra, dec, target, propid, image, instrument
    instrument is not very useful, only used in aladin tables and can be
    passed as arg
    """
    data = pd.DataFrame()
    with open(filename, 'r') as inp:
        lines = inp.readlines()

    if lines[0].startswith('filename'):
        lines = lines[1:]

    radecs = np.array([np.array(l.strip().split('point')[1].split(), dtype=float) for l in lines])
    s_region = np.array([l.strip().split(';')[1].strip() for l in lines])
    pids, targets = zip(*[l.split(';')[0].split('/')[0].split('_') for l in lines])

    data['filename'] = [l.split(';')[0].replace('_footprint_ds9_linear.reg', '.fits') for l in lines]
    data['ra'] = radecs.T[0]
    data['dec'] = radecs.T[1]
    data['target'] = list(targets)
    data['propid'] = list(pids)
    data['image'] = [l.split(';')[0].split('/')[-1].split('_')[0] for l in lines]
    data['instrument'] = instrument
    if instrument is None:
        data['instrument'] = [l.split(';')[0].split('/')[0] for l in radec]

    return data, s_region

def parse_poly(line, closed=True, return_string=False):
    """Convert a polygon into N,2 np array. If closed, repeat first coords at end."""
    repd = {'J2000 ': '', 'GSC1 ': '', 'ICRS ': '', 'multi': '', 'polygon': '', ')': '', '(': ''}
    line = replace_all(line.lower(), repd).split('#')[0]
    try:
        # ds9-like format all values separated by ,
        polyline = np.array(line.strip().split(','), dtype=float)
    except:
        # shapely-like format (x0 y0, x1 y1, ...)
        line = line.strip().replace(' ',',').replace(',,',',')
        polyline = np.array(line.strip().split(','), dtype=float)

    if closed:
        if False in polyline[:2] == polyline[-2:]:
            polyline = np.append(polyline, polyline[:2])

    retv  = polyline.reshape(len(polyline)/2, 2)

    if return_string:
        retv = ','.join(['[{:.6f}, {:.6f}]'.format(*c) for c in retv])

    return retv

def center_from_simbad(target):
    """Query Simbad for the coordinates of a target."""
    from astroquery.simbad import Simbad
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    def sstr(attr):
        """
        Strip the value from a simbad table string
        e.g.
        >>> str(q['RA'])
        >>> '     RA     \n  "h:m:s"   \n------------\n04 52 25.040'
        >>> sstr(q['RA'])
        >>> '04 52 25.040'
        """
        return str(attr).split('\n')[-1]

    q = Simbad.query_object(target)

    if q is None:
        print('Error, can not query simbad for {}'.format(target))
        return np.nan, np.nan

    rd = SkyCoord(ra=sstr(q['RA']), dec=sstr(q['DEC']),
                  unit=(u.hourangle, u.deg))
    return rd.ra.value, rd.dec.value


def main(percent_tolerance=49.):
    """
    Check for polygon intersections.
    Code to plot if needed to re-work this:
    #for k, p in enumerate([p1, p2]):
    #    p_ = parse_poly(line, closed=False)
    #    ax.plot(p_[:, 0], p_[:, 1])
    """
    if not os.path.isfile('all_poly_simp.csv'):
        print('simple polygon region file not found. Trying to build.')
        err = os.system('ls */*lin*reg > tmp')
        if err == 0:
            fnames = [f.strip() for f in open('tmp').readlines()]
            os.system('rm tmp')
            simplify_footprints(fnames)
        else:
            print('failed to build all_poly_simp.csv, run simplify_footprints')
            sys.exit()

    data, s_region = read_footprints('all_poly_simp.csv', instrument='multiband')
    coords = [parse_poly(p) for p in s_region]
    polys = np.array([Polygon(c) for c in coords])
    not_overlapping = False

    allpaths = [os.path.split(f)[0] for f in data['filename']]
    paths = np.unique(allpaths)
    targ_olap = []
    tol_olap = []
    for l, path in enumerate(paths):
        inds = [k for k, a in enumerate(data['filename']) if path == os.path.split(a)[0]]
        for i, j in itertools.product(inds, inds):
            if i <= j:
                continue
            p1 = polys[i]
            p2 = polys[j]
            t1 = data['target'].iloc[i]
            t2 = data['target'].iloc[j]
            im1 = data['filename'].iloc[i]
            im2 = data['filename'].iloc[j]
            if p1.intersects(p2):
                p3 = p1.intersection(p2)
                olap = p3.area / p1.area * 100
                #print('{} intersects with {} {} {} by {:.2f}'.format(t1, t2, i, j, olap))
                # Intersection but targets do not match
                if t1 != t2:
                    print('{} should not intersect with {} {} {} by {:.2f}%'.format(t1, t2, i, j, olap))
                    targ_olap.append(i)
                    targ_olap.append(j)
                else:
                    # Intersection, targets match, but not enough overlap
                    if olap <= percent_tolerance:
                        print('{} should not intersect with {} {} {} by {:.2f}%'.format(t1, t2, i, j, olap))
                        tol_olap.append(i)
                        tol_olap.append(j)
            else:
                if t1 == t2:
                    # No intersection but targets match.
                    print('{} does not intersect with {} {} {}'.format(im1, im2, i, j))
                    not_overlapping = True

    if len(targ_olap) > 1:
        idx = np.unique(targ_olap)
        ps = polys[idx]
        groups, gidx = group_polygons(ps, return_index=True)
        if len(gidx) > 1:
            for i in range(len(gidx)):
                df = data.iloc[idx[gidx[i]]].copy(deep=True)
                df['group'] = i
                df['s_region'] = s_region[idx[gidx[i]]]
                df_olap = df_olap.append(df, ignore_index=True)
        df_olap.to_csv('mismatched_target.csv', index=False, sep=';')
        fix_overlaps(fname='mismatched_target.csv')

    if len(tol_olap) > 1:
        df_olap = pd.DataFrame()
        idx = np.unique(tol_olap)
        ps = polys[idx]
        groups, gidx = group_polygons(ps, return_index=True)
        if len(gidx) > 1:
            for i in range(len(gidx)):
                df = data.iloc[idx[gidx[i]]].copy(deep=True)
                df['group'] = i
                df['s_region'] = s_region[idx[gidx[i]]]
                df_olap = df_olap.append(df, ignore_index=True)
        df_olap.to_csv('beyond_tolerance.csv', index=False, sep=';')
        fix_overlaps(fname='beyond_tolerance.csv')

    if not_overlapping:
        notoverlapping()
        fix_overlaps(test_file=True)
    return

def split_polygons(polygonlist, tol=49.):
    """
    Return list of polygons that intersect with the first value and
    a list of polygons that do not interesect with the first value.
    """
    ins = []
    outs = []
    if len(polygonlist) > 0:
        p0 = polygonlist[0]
        ins.append(p0)
        for i in range(len(polygonlist)-1):
            p = polygonlist[i+1]
            if p0.intersects(p):
                olap = p0.intersection(p).area / p.area * 100
                if olap > tol:
                    ins.append(p)
                else:
                    print(olap)
                    outs.append(p)
            else:
                outs.append(p)
    return ins, outs



def group_polygons(polylist, return_index=False):
    """
    group a list of Polygons into ones that intersect with eachother
    option to return the index of the origional list
    """
    npolys = len(polylist)
    outs = polylist
    groups = []
    while len(outs) != 0:
        ins, outs = split_polygons(outs)
        if len(ins) > 0:
            groups.append(ins)
    assert npolys == np.sum([len(g) for g in groups]), 'lost polygons'
    retv = groups
    if return_index:
        inds = [np.concatenate([[i for (i, p) in enumerate(polylist) if p == g]
                                 for g in group]) for group in groups]
        retv = (groups, inds)
    return retv


def notoverlapping():
    """
    write a csv of fields in the same directory that do not overlap.
    """
    data, s_region = read_footprints('all_poly_simp.csv', instrument='multiband')
    coords = [parse_poly(p) for p in s_region]
    polys = np.array([Polygon(c) for c in coords])
    # may need to have a figure for each if statement...
    allpaths = [os.path.split(f)[0] for f in data['filename']]
    paths = np.unique(allpaths)
    not_overlapping = pd.DataFrame()
    for path in paths:
        inds = np.array([k for k, a in enumerate(data['filename'])
                         if path == os.path.split(a)[0]])
        groups, gidx = group_polygons(polys[inds], return_index=True)
        if len(gidx) > 1:
            for i in range(len(gidx)):
                df = data.iloc[inds[gidx[i]]].copy(deep=True)
                df['group'] = i
                df['s_region'] = s_region[inds[gidx[i]]]
                not_overlapping = not_overlapping.append(df, ignore_index=True)
        else:
            #print('all in {} overlap'.format(path))
            pass
    not_overlapping.to_csv('not_overlapping.csv', index=False, sep=';')


def fix_overlaps(fname='not_overlapping.csv', test_file=False):
    """
    use not_overlapping.csv to group polygons within each directory and print
    bash commands to move the files to temporary directories.
    test_file will output the region coords as if the moves were made to use
    in aladin_tables to visualize changes.
    """
    try:
        data = pd.read_csv(fname, sep=';')
    except:
        print('Probably nothing to fix:')
        print(sys.exc_info()[1])
        sys.exit()

    if test_file:
        testfix = pd.DataFrame()

    # If there are more than 7 groups of polygons perhaps an automated search
    # is not what you need...
    order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    paths, inds = np.unique([os.path.split(f)[0] for f in data['filename']],
                             return_index=True)

    inds = np.append(inds, len(data))

    for i in range(len(inds)-1):
        idx = np.arange(inds[i], inds[i+1])
        groups, ginds = np.unique(data.iloc[idx]['group'], return_index=True)
        ginds = np.append(ginds, len(idx)-1)
        for j in range(len(ginds)-1):
            gidx = np.arange(idx[ginds[j]], idx[ginds[j+1]])
            if j == len(ginds) - 2:
                if len(gidx) == 1:
                    gidx = np.append(gidx, gidx + 1)
                else:
                    gidx = np.append(gidx, idx[-1])
            #print gidx
            #print(data['filename'].iloc[gidx])
            newdir = '-'.join([paths[i], order[j]])
            print('mkdir {}'.format(newdir))
            print('mv {} {}'.format(' '.join(data['filename'].iloc[gidx]), newdir))
            if test_file:
                newfnames = [os.path.join(newdir, os.path.split(d)[1])
                             for d in data['filename'].iloc[gidx]]
                df = data.iloc[gidx].copy(deep=True)
                df['filename'] = newfnames
                testfix = testfix.append(df, ignore_index=True)

    if test_file:
        testfix.to_csv('test_not_overlapping.csv', index=False, sep=';')
    return

if __name__ == "__main__":
    import pdb; pdb.set_trace()
    main()
