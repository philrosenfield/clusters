import argparse
from palettable.colorbrewer import qualitative
import numpy as np
import sys

import string
allTheLetters = string.lowercase

def replace_all(text, dic):
    """perfrom text.replace(key, value) for all keys and values in dic"""
    for old, new in dic.iteritems():
        text = text.replace(old, new)
    return text


def read_hlacsv(filename, maxlength=1200, raunit='decimal'):
    """
    Read csv output from MAST discovery portal.
    First line is column format
    Second line is column names

    NOTE: In experience the max length of the s_region column is 1064 chars
    There is NO check to make sure this line is not tuncated
    (currently assumed max is 1200 chars)
    """
    inp = open(filename, 'r')
    lines = inp.readlines()

    idx, comlines = zip(*[(i, l.strip()) for (i, l) in enumerate(lines)
        if l.startswith('#')])
    colfmtline, = [l for l in comlines if l.startswith('#@')]
    colfmts = colfmtline.translate(None, '#@ ').strip()
    ihead = idx[-1]+1
    colnames = lines[ihead].split(',')
    converters=None
    if 'obs_title' in colnames:
        print('WARNING Observation Title is in catalog. '
              'There could be commas in the field which will break the data reader')
        converters = {9: lambda s: s.replace("'", "&#8217;")}
    repd = {'ra': 'float', 'dec': 'float', 'string': '|S{}'.format(maxlength),
            'float': '<f8', 'int': '<f8'}

    if raunit != 'decimal':
        repd['ra'] = '|S{}'.format(maxlength)
        repd['dec'] = '|S{}'.format(maxlength)

    fmts = replace_all(colfmts, repd).split(',')

    dtype = [(c, f) for c, f in zip(colnames, fmts)]
    # the converter is for proposal last name, errors occur with O'Connell etc
    # could sub in the
    data = np.genfromtxt(filename, dtype=dtype, delimiter=',', skip_header=ihead + 1,
                         converters=converters)
    return data


def polygon_line(name, polygon_array, color='#ee2345', lw=3):
    """
    add polygons
    some lines in the data have two polygons (each chip)
    """
    def grab_coors(line):
        coords = ', '.join(['[{}, {}]'.format(j, k)
                           for (j, k) in zip(line[::2], line[1::2])])
        return coords

    head = "var {0} = A.graphicOverlay({{color: \'{1}\', lineWidth: {2}}});\naladin.addOverlay({0});\n".format(name, color, lw)

    poly_str = []

    for line in polygon_array:
        # LAZY: could use astropy to convert coord systems
        repd = {'J2000 ': '', 'GSC1 ': '', 'ICRS ': ''}
        poly_line = replace_all(line, repd).split('POLYGON ')[1:]
        if len(poly_line) > 1:
            for line in poly_line:
                coords = grab_coors(line.split())
                poly_str.append('A.polygon([{}])'.format(coords))
        else:
            #import pdb; pdb.set_trace()
            coords = grab_coors(poly_line[0].split())
            poly_str.append('A.polygon([{}])'.format(coords))

    polygons = ', '.join(poly_str)

    return ''.join((head, '{0}.addFootprints([{1}]);\n'.format(name, polygons)))

def catalog_line(name, data, ms=10, color='red'):
    """
    Add  markers and popup
    """
    head = ("var {0} = A.catalog({{name: '{0}', sourceSize: {1}, color: '{2}'}});\n"
                "aladin.addCatalog({0});\n".format(name, ms, color))
    fmt = ("A.marker(%(s_ra)f, %(s_dec)f, "
           "{popupTitle: '%(target_name)s', "
            "popupDesc: "
               "'<em>Instrument:</em> %(instrument)s "
               "<em>Filters:</em> %(filters)s <br/>"
               "<em>PI:</em> %(proposal_pi)s <em>PID:</em> %(proposal_id)s <br/>"
               "<em>Exp time:</em> %(t_exptime)i <br/>"
               "<br/><a href=\"%(jpegURL)s\" target=\"_blank\">jpeg preview</a>'})")
    catalog = ', '.join([fmt % d for d in data])
    cat_line = "{0}.addSources([{1}]);\n".format(name, catalog)
    return ''.join((head, cat_line))



header = \
"""
<html>
<head>
<link rel='stylesheet' href='http://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.css' />
<script type='text/javascript' src='http://code.jquery.com/jquery-1.9.1.min.js' charset='utf-8'></script>

<div id='aladin-lite-div'></div>
<script type='text/javascript' src='http://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.js' charset='utf-8'></script>
</head>

<script>
var aladin = A.aladin('#aladin-lite-div', {target: '03 46 45.6 -74 26 40', fov: 30.0, fullScreen: true});
"""

footer = '</script>\n</html>\n'

def make_html(outfile=None, csvs=None, poly_names=None, poly_colors=None, ms=10,
              lw=3, cat_colors=None, cat_names=None):

    if csvs is None:
        print('Where is the data?')
        sys.exit(1)

    ncsvs = len(csvs)
    if poly_names is None:
        poly_names = ['overlay{}'.format(allTheLetters[i]) for i in range(ncsvs)]

    if cat_names is None:
        cat_names = ['hstcat{}'.format(allTheLetters[i]) for i in range(ncsvs)]

    if poly_colors is None:
        try:
            ncol = np.max((3, ncsvs))
            poly_colors = qualitative.__getattribute__('Paired_{}'.format(ncol)).hex_colors
        except KeyError:
            print('too many csvs!')
            some_poly_colors = qualitative.Paired['max'].hex_colors
            # cycle... could probably make it a generator...
            poly_colors = some_poly_colors * 50

    if cat_colors is None:
        cat_colors = poly_colors

    pstr = [header]
    for i in range(ncsvs):
        data = read_hlacsv(csvs[i])
        pstr.append(polygon_line(poly_names[i], data['s_region'], lw=lw,
                                 color=poly_colors[i]))
        pstr.append(catalog_line(cat_names[i], data, ms=ms,
                                 color=cat_colors[i]))
    pstr.append(footer)
    with open(outfile, 'w') as out:
        [out.write(p) for p in pstr]
    return

def main(argv):
    parser = argparse.ArgumentParser(description="Create aladin view from MAST \
                                                  DiscoveryPortal csv output")

    parser.add_argument('-o', '--output', type=str, default='script.html',
                        help='name of output file')

    parser.add_argument('name', type=str, nargs='*',
                        help='name(s) of csv from MAST DiscoveryPortal')

    args = parser.parse_args(argv)

    make_html(outfile=args.output, csvs=args.name)

if __name__ == '__main__':
    main(sys.argv[1:])
