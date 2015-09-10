from bokeh.plotting import ColumnDataSource, figure, gridplot, output_file, show
from bokeh.embed import file_html, components
from bokeh.resources import CDN
from bokeh.models import Range1d
import os
from pandas import *


def bplot_cmd_xy(photfile, xyfile):

    pid, target, filter1, filter2 = \
        os.path.split(photfile)[1].split('.gst')[0].split('_')

    title = '{} {}'.format(pid, target)
    xlabel = '{}-{}'.format(filter1, filter2)
    ylabel = '{}'.format(filter2)
    outfile = os.path.split(xyfile)[1].replace('.xy', '') + '.html'

    mag1, mag2 = np.genfromtxt(photfile, unpack=True)
    _, _, x, y =np.genfromtxt(xyfile, unpack=True)
    color = mag1 - mag2
    good, = np.nonzero((color > -0.5) & (mag2 < 25.5) & (color < 4))
    data = np.column_stack((color[good], mag2[good], x[good], y[good]))
    
    df = DataFrame(data, columns=['color', 'mag2', 'x', 'y'])    
    source  = ColumnDataSource(df)

    TOOLS = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,save"
    plot_config = dict(plot_width=500, plot_height=500, tools=TOOLS)
    cs = ['#8C3B49', '#212053']

    p1 = figure(title=title, **plot_config)
    p1.circle("color", "mag2", size=1, source=source, color=cs[0])
    p1.y_range = Range1d(26, 14)
    p1.yaxis.axis_label = ylabel
    p1.xaxis.axis_label = xlabel

    p2 = figure(**plot_config)
    p2.circle(data[:, 2], data[:, 3], size=1, source=source, color=cs[1])
    p2.yaxis.axis_label = "Y"
    p2.xaxis.axis_label = "X"
    
    p = gridplot([[p1, p2]])
    
    html = file_html(p, CDN, title)
    
    with open(outfile, 'w') as f:
        f.write(html)
    
    #script, div = components(p)
    