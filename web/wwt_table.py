import argparse
import numpy as np
import sys
import os
import pandas as pd
from fitshelper.fitshelper.footprints import parse_poly


def read_MAST(filename):
    """
    Read csv output from MAST discovery portal.
    First line is column format
    Second line is column names
    """
    data = pd.read_csv(filename,  header=0, skiprows=[0])
    return data


def polygon_line(polygon_array, color='#ee2345', lw=3, ind=0, alpha=1, label=''):
    """
    add polygons
    some lines in the data have two polygons (each chip)
    """
    pstr = ','.join(['[{}, {}]'.format(*p) for p in polygon_array])
    name = 'poly{0:d}'.format(ind)
    d = {'ind': ind, 'color': color, 'lw': lw, 'alpha': alpha, 'name': name,
         'pstr': pstr, 'label': label}
    pline = ("var points{ind:d} = [{pstr:s}];\n"
             "{name:s} = createWWTPolyLine(\"{color:s}\", {lw:d}, {alpha:f}, points{ind:d}, \"{label:s}\");\n"
             "{name:s}.set_id({name:s});").format(**d)

    return pline, name

def get_labels(data):
    labels = []
    for i in range(len(data)):
        try:
            filters = data.filters.loc[i]
        except:
            filter1 = data.filter1.loc[i]
            filter2 = data.filter2.loc[i]
            filters = ','.join([f for f in [filter1, filter2] if not '...' in f])

        labels.append('{:s} {:s} {:s}'.format(data.target_name.loc[i],
                                              data.instrument_name.loc[i],
                                              filters))
    return labels

def polygon_lines(data, plykw=None):
    """
    Add markers and popup
    """
    plykw = plykw or {}

    polys = [parse_poly(p) for p in data.s_region]
    labels = get_labels(data)
    import pdb; pdb.set_trace()
    plines, names = zip(*[polygon_line(polys[i], ind=i, label=labels[i])
                          for i in range(len(polys))])
    return '\n'.join(plines), list(names)

src = os.path.split(__file__)[0]
# target ra dec is set to near the to view both the LMC and SMC
with open(os.path.join(src, 'header.html')) as inp:
    header = inp.readlines()
header = '\n'.join(header)

with open(os.path.join(src, 'footer.html')) as inp:
    footer = inp.readlines()
footer = '\n'.join(footer)

def make_html(csv, outfile=None, plykw=None):
    plykw = plykw or {}
    plines, names = polygon_lines(read_MAST(csv), plykw=plykw)

    outfile = outfile or csv.replace('.csv', '_wwt.html')
    html = header
    html += """
        <script>
         // Create the WorldWide telescope object variable
         var wwt;
         // Create variables to hold some annotation objects
         """
    html += '\n'.join(['var {0:s};'.format(n) for n in names])
    wwtadd = '\n'.join(['wwt.addAnnotation({0:s});'.format(n) for n in names])
    wwtrm = '\n'.join(['wwt.removeAnnotation({0:s});'.format(n) for n in names])
    html += """
        // Create variables to hold the changeable settings
        var bShowCrosshairs = true;
        var bShowUI = false;
        var bShowFigures = true;
        var bShowAnnotations = true;
        var bShowCircles = false;
        var bShowPolygon = false;
        var bShowPolyLine = false;

        // A simple function to toggle the settings
        // This function is called from the checkbox entries setup in the html table
         function toggleSetting(text)
        {
                switch (text)
            {
                    case 'ShowUI':
                        bShowUI = !bShowUI;
                        wwt.hideUI(!bShowUI);
                        break;
                     case 'ShowCrosshairs':
                        bShowCrosshairs = !bShowCrosshairs;
                        wwt.settings.set_showCrosshairs(bShowCrosshairs);
                        break;
                     case 'ShowFigures':
                        bShowFigures = !bShowFigures;
                        wwt.settings.set_showConstellationFigures(bShowFigures);
                        break;
                     case 'ShowPolyLine':
                        bShowPolyLine = !bShowPolyLine;
                        if (bShowPolyLine) {
                        """
    html += wwtadd
    html += """
                        } else {
    """
    html += wwtrm
    html += """
                        }
                        break;
                }
        }
         // A function to change the view to different constellations
        // Note the "instant" parameter set to false for smooth slewing
        // This function is called from the button entries in the html table
         function GotoConstellation(text) {

            switch (text) {
                case 'Sagittarius':
                    wwt.gotoRaDecZoom(286.485, -27.5231666666667, 60, false);
                    break;
                 case 'Aquarius':
                    wwt.gotoRaDecZoom(334.345, -9.21083333333333, 60, false);
                    break;
            }
        }
         // A function to create a polygon

        function createWWTPolygon(fill, lineColor, fillColor, lineWidth, opacity, points) {
            var poly = wwt.createPolygon(fill);
            poly.set_lineColor(lineColor);
            poly.set_fillColor(fillColor);
            poly.set_lineWidth(lineWidth);
            poly.set_opacity(opacity);
            for (var i in points) {
                poly.addPoint(points[i][0], points[i][1]);
            }
            return poly;
        }

        // A function to create a polyline object
         function createWWTPolyLine(lineColor, lineWidth, opacity, points, label) {
            var polyline = wwt.createPolyLine(true);
            polyline.set_lineColor(lineColor);
            polyline.set_lineWidth(lineWidth);
            polyline.set_opacity(opacity);
            if (label) {
                polyline.set_label(label);
                polyline.set_showHoverLabel(true);
            }
            for (var i in points) {
                polyline.addPoint(points[i][0], points[i][1]);
            }

            return polyline;
        }
            // A function to create a circle
        function createWWTCircle(fill, lineColor, fillColor, lineWidth, opacity, radius, skyRelative, ra, dec) {
            var circle = wwt.createCircle(fill);
            circle.set_lineColor(lineColor);
            circle.set_fillColor(fillColor);
            circle.set_lineWidth(lineWidth);
            circle.set_opacity(opacity);
            circle.set_radius(radius);
            circle.set_skyRelative(skyRelative);
            circle.setCenter(ra, dec);
            return circle;
        }
           function initialize() {
              wwt = wwtlib.WWTControl.initControl("WWTCanvas");
              wwt.add_ready(wwtReady);
          }
            function wwtReady() {
                wwt.settings.set_showCrosshairs(bShowCrosshairs);
                wwt.settings.set_showConstellationFigures(bShowFigures);
                wwt.hideUI(!bShowUI);
                wwt.settings.set_showConstellationBoundries(true);
                wwt.gotoRaDecZoom(56.69, -74.44, 30, true);
    """
    html += plines
    html += """    }
         </script>
         </head>
         """
    html += footer
    with open(outfile, 'w') as out:
        [out.write(l) for l in html]
    return

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create a WWT view from MAST \
                                                  DiscoveryPortal csv output")

    parser.add_argument('-o', '--output', type=str, help='name of output file')

    parser.add_argument('-v', '--pdb', action='store_true', help='invoke pdb')

    parser.add_argument('name', type=str, help='name of csv')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.pdb:
        import pdb
        pdb.set_trace()

    make_html(args.name, outfile=args.output)

if __name__ == '__main__':
    sys.exit(main())
