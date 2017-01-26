import argparse
import numpy as np
import sys

import pandas as pd
import string
from ..utils import replace_all
from fitshelper.footprints import parse_poly
import seaborn
seaborn.set()
allTheLetters = string.ascii_lowercase


def read_hlacsv(filename):
    """
    Read csv output from MAST discovery portal.
    First line is column format
    Second line is column names
    """
    import os
    data = pd.read_csv(filename, sep=';')
    if not 'target' in data.columns:
        data['target'] = [os.path.split(f)[0].split('_')[1] for f in data['filename']]

    if not 'pid' in data.columns:
        data['propid'] = [os.path.split(f)[0].split('_')[0] for f in data['filename']]

    if not 'ra' in data.keys():
        radecs = np.array([data['radec'].iloc[i].replace('point','').split()
                           for i in range(len(data))], dtype=float)
        data['ra'] = radecs.T[0]
        data['dec'] = radecs.T[1]
    return data

def read_csv(filename):
    """
    Read csv output from MAST discovery portal.
    First line is column format
    Second line is column names
    """
    data = pd.DataFrame()
    with open(filename, 'r') as inp:
        lines = inp.readlines()

    radec = [l.strip() for l in lines if 'point' in l]
    radecs = np.array([np.array(l.split(',')[1].split('#')[0].replace('point','').split(),
                                dtype=float) for l in radec])
    targets, pids = zip(*[l.split(',')[0].split('/')[1].split('_') for l in radec])

    data['ra'] = radecs.T[0]
    data['dec'] = radecs.T[1]
    data['target'] = list(targets)
    data['propid'] = list(pids)
    data['instrument'] = [l.split(',')[0].split('/')[0] for l in radec]
    s_region = [l.strip().split('reg')[1] for l in lines if 'polygon' in l]
    return data, s_region


def polygon_line(name, polygon_array, color='#ee2345', lw=3):
    """
    add polygons
    some lines in the data have two polygons (each chip)
    """
    head = "var {0} = A.graphicOverlay({{color: \'{1}\', lineWidth: {2}}});\naladin.addOverlay({0});\n".format(name, color, lw)

    poly_str = [('A.polygon([{}])'.format(parse_poly(line, return_string=True)))
                for line in polygon_array]

    polygons = ', '.join(poly_str)

    return ''.join((head, '{0}.addFootprints([{1}]);\n'.format(name, polygons)))

def catalog_line(name, data, ms=10, color='red'):
    """
    Add markers and popup
    """
    imgfmt = '<img src=\"cmds/{0}\" width=\"100%\" alt=\"{0}\">'
    #imgfmt = '<iframe src=\"{0}\" width=\"100%\" height=\"100%\"></iframe>'
    if 'images' in data.columns:
        data['img'] = pd.Series()
        for i in range(len(data)):
            if not isinstance(data['images'].iloc[i], float):
                images = data['images'].iloc[i].split(',')
                if len(images) > 1:
                    data['img'].iloc[i] = '<br/>'.join([imgfmt.format(img) for img in images])
                else:
                    data['img'].iloc[i] = imgfmt.format(images[0])
            else:
                data['img'].iloc[i] = "None"

    head = ("var {0} = A.catalog({{name: '{0}', sourceSize: {1}, color: '{2}'}});\n"
            "aladin.addCatalog({0});\n".format(name, ms, color))

    #fmt = ("A.source(%(ra)f, %(dec)f, "
    #       "{name: '%(target)s', "
    #       "image: "
    #       "'%(img)s'})")

    fmt = ("A.marker(%(ra)s, %(dec)s, {popupTitle: '%(target)s', popupDesc: '<em>Filters:</em> \"%(filters)s\"'})")
    catalog = ', '.join([fmt % data.iloc[d] for d in range(len(data))])
    cat_line = "{0}{1}.addSources([{2}]);\n".format(head, name, catalog)
    return cat_line

# target ra dec is set to near the to view both the LMC and SMC
header = \
"""
<html>
<head>
<link rel="stylesheet" href="stylesheets/styles.css">
<link rel="stylesheet" href="stylesheets/github-light.css">
<link rel='stylesheet' href='http://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.css' />
<script type='text/javascript' src='http://code.jquery.com/jquery-1.9.1.min.js' charset='utf-8'></script>



<table>
<tr><td height='500px' width='500px'><div id='aladin-lite-div' class='side_div'></div></td>
<td><div id='infoDiv' class='side_div'>&nbsp; </div></td></tr>
</table>
<script type='text/javascript' src='http://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.js' charset='utf-8'></script></head>

<script>
var aladin = A.aladin('#aladin-lite-div', {target: '03 46 45.6 -74 26 40', fov: 30.0, fullScreen: false});

aladin.addCatalog(A.catalogFromVizieR('J/MNRAS/430/676/table2', '03 46 46.5 -74 26 40', 30.0, {onClick: 'showTable', name: 'Baumgardt2013'}));
"""
# aladin.addCatalog(A.catalogFromVizieR('J/MNRAS/389/678/table3', '03 46 46.5 -74 26 40', 30.0, {onClick: 'showTable', name: 'Bica2006'}));
# aladin.addCatalog(A.catalogFromVizieR('J/A+A/517/A50/clusters', '03 46 46.5 -74 26 40', 30.0, {onClick: 'showTable', name: 'Glatt2010'}));

#// define function triggered when  a source is hovered
#aladin.on('objectHovered', function(object) {
#    var msg;
#    if (object) {
#        msg = 'You hovered object ' + object.data.name + ' located at ' + object.ra + ', ' + object.dec;
#    }
#    else {
#        msg = 'No object hovered';
#    }
#    $('#infoDiv').html(msg);
#});

footer = """
// define function triggered when an object is clicked
var objClicked;
aladin.on('objectClicked', function(object) {
    var msg;
    if (object) {
        objClicked = object;
      object.select();
        // msg = object.data.name + ' ' + object.ra + ', ' + object.dec + object.data.image;
        msg = '<h2>' + object.data.name + '</h2>' + object.data.image;
    }
    else {
        msg = '';
        objClicked.deselect();
    }
    $('#infoDiv').html(msg);
});
</script>
</html>
"""

def make_html(outfile=None, csvs=None, poly_names=None, poly_colors=None, ms=10,
              lw=3, cat_colors=None, cat_names=None):

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
        try:
            data = read_hlacsv(csvs[i])
            data['instrument'] = 'multiband'
            s_region = data['s_region']
        except:
            data, s_region = read_csv(csvs[i])
            # try:
                # data, s_region = read_footprints(csvs[i], instrument='WFC3')
            # except:

        pstr.append(polygon_line(poly_names[i], s_region, lw=lw))

        pstr.append(catalog_line(cat_names[i], data, ms=ms))

    pstr.append(footer)
    with open(outfile, 'w') as out:
        [out.write(p) for p in pstr]
    return

def main(argv):
    parser = argparse.ArgumentParser(description="Create aladin view from MAST \
                                                  DiscoveryPortal csv output")

    parser.add_argument('-o', '--output', type=str, default='default',
                        help='name of output file')

    parser.add_argument('-v', '--pdb', action='store_true',
                        help='invoke pdb')

    parser.add_argument('name', type=str, nargs='*',
                        help='name(s) of csv(s)')

    args = parser.parse_args(argv)

    if args.pdb:
        import pdb
        pdb.set_trace()
    if args.output == 'default':
        if len(args.name) > 1:
            args.output = 'script.html'
        else:
            args.output = args.name[0].replace('csv', 'html')

    make_html(outfile=args.output, csvs=args.name)

if __name__ == '__main__':
    main(sys.argv[1:])
