import argparse
import numpy as np
import sys
import pandas as pd
from astropy.table import Table
import string
allTheLetters = string.ascii_lowercase

def replace_all(text, dic):
    """perfrom text.replace(key, value) for all keys and values in dic"""
    if text is None:
        return text
    for old, new in dic.items():
        text = text.replace(old, new)
    return text


def read_hlacsv(filename, maxlength=1200, raunit='decimal'):
    """
    Read csv output from MAST discovery portal.
    First line is column format
    Second line is column names
    """
    return pd.read_csv(filename, header=0)


def polygon_line(name, polygon_array, color='#ee2345', lw=3):
    """
    add polygons
    some lines in the data have two polygons (each chip)
    """
    def grab_coors(line):
        coords = ', '.join(['[{:.6f}, {:.6f}]'.format(float(j), float(k))
                           for (j, k) in zip(line[::2], line[1::2])])
        return coords

    head = "var {0} = A.graphicOverlay({{color: \'{1}\', lineWidth: {2}}});\naladin.addOverlay({0});\n".format(name, color, lw)

    poly_str = []

    for i in range(len(polygon_array)):
        line = polygon_array[i]
        if line is np.nan or len(line) == 0:
            continue
        line = line.replace('\'', '')
        # LAZY: could use astropy to convert coord systems
        repd = {'j2000 ': '', 'gsc1 ': '', 'icrs ': '', 'other': ''}
        poly_line = replace_all(line, repd).split('POLYGON ')[1:]
        if len(poly_line) > 1:
            for line in poly_line:
                coords = grab_coors(line.split())
                poly_str.append('A.polygon([{}])'.format(coords))
        elif len(poly_line) == 1:
            coords = grab_coors(poly_line[0].split())
            poly_str.append('A.polygon([{}])'.format(coords))


    polygons = ', '.join(poly_str)

    return ''.join((head, '{0}.addFootprints([{1}]);\n'.format(name, polygons)))

def catalog_line(name, data, ms=10, color='red', mast=True):
    """
    Add markers and popup
    """
    head = ("var {0} = A.catalog({{name: '{0}', sourceSize: {1}, color: '{2}'}});\n"
            "aladin.addCatalog({0});\n".format(name, ms, color))
    if mast:
        fmt = ("A.marker(%(ra)f, %(dec)f, "
               "{popupTitle: '%(target)s', "
               "popupDesc: "
               "'<em>Instrument:</em> %(instrument)s "
               "<em>Filters:</em> %(filter1)s, %(filter2)s  <br/>"
               "<em>PI:</em> %(pr_inv)s <em>PID:</em> %(propid)s <br/>"
               "<em>Exp time:</em> %(exptime)i <br/>'})")
               #"<br/><a href=\"%(jpegURL)s\" target=\"_blank\"><img src=\"%(jpegURL)s\" alt=\"%(target_name)s jpeg preview\"></a>'})")
    else:
        fmt = ("A.marker(%(ra)s, %(dec)s, "
               "{popupTitle: '%(propid)s %(target)s', "
                "popupDesc: "
                "'<em>Instrument:</em> %(instrument)s-%(detector)s"
                "<em>Filters:</em> %(filter1)s, %(filter2)s <br/>"
                "<em>PI:</em> %(pr_inv)s <br/>"
                "<em>Exp time:</em> %(exptime)i <br/>"
                "\"%(filename)s\"<br/>'})")
    catalog = ', '.join([fmt % data.iloc[d] for d in range(len(data))])
    cat_line = "{0}{1}.addSources([{2}]);\n".format(head, name, catalog)
    return cat_line

# target ra dec is set to near the to view both the LMC and SMC
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
aladin.setFOVRange(9, 70)
aladin.addCatalog(A.catalogFromVizieR('J/MNRAS/389/678/table3', '03 46 46.5 -74 26 40', 20.0, {onClick: 'showTable', name: 'Bica2006'}));
aladin.addCatalog(A.catalogFromVizieR('J/A+A/517/A50/clusters', '03 46 46.5 -74 26 40', 20.0, {onClick: 'showTable', name: 'Glatt2010'}));
aladin.addCatalog(A.catalogFromVizieR('J/MNRAS/430/676/table2', '03 46 46.5 -74 26 40', 20.0, {onClick: 'showTable', name: 'Baumgardt2013'}));
"""

footer = '</script>\n</html>\n'

def make_html(outfile=None, csvs=None, poly_names=None, poly_colors=None, ms=10,
              lw=3, cat_colors=None, cat_names=None, mast=True):

    if csvs is None:
        print('Where are the data?')
        sys.exit(1)

    ncsvs = len(csvs)
    if poly_names is None:
        poly_names = ['overlay{}'.format(allTheLetters[i]) for i in range(ncsvs)]

    if cat_names is None:
        cat_names = ['hstcat{}'.format(allTheLetters[i]) for i in range(ncsvs)]

    pstr = [header]
    for i in range(ncsvs):
        if mast:
            data = read_hlacsv(csvs[i])
            pstr.append(polygon_line(poly_names[i], data['s_region'], lw=lw,
                                     color='g'))
        else:
            data = Table.read(csvs[i], header_start=0, delimiter=' ')

        pstr.append(catalog_line(cat_names[i], data, ms=ms, mast=mast,
                                 color='r'))

    pstr.append(footer)
    with open(outfile, 'w') as out:
        [out.write(p) for p in pstr]
    return

def main(argv):
    parser = argparse.ArgumentParser(description="Create aladin view from MAST \
                                                  DiscoveryPortal csv output")

    parser.add_argument('-o', '--output', type=str, default='script.html',
                        help='name of output file')

    parser.add_argument('-m', '--mast', action='store_false',
                        help='csv is not from MAST (no s_regions, html link, etc)')

    parser.add_argument('name', type=str, nargs='*',
                        help='name(s) of csv(s)')

    args = parser.parse_args(argv)

    make_html(outfile=args.output, csvs=args.name, mast=args.mast)

if __name__ == '__main__':
    main(sys.argv[1:])
