import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from match.scripts.utils import parse_pipeline
from match.scripts import asts
from clusters.data_plots import cmd


def getfakes():
    loc = '/Users/rosenfield/research/clusters/asteca/acs_wfc3/match_runs'
    fakes = [os.path.join(loc, f)
             for f in ['9891_NGC1978_F555W_F814W.gst.matchfake',
                       '12257_HODGE2_F475W-F814W.gst.matchfake',
                       '12257_NGC1718_F475W-F814W.gst.matchfake',
                       '12257_NGC2173_F475W-F814W.gst.matchfake',
                       '12257_NGC2203_F475W-F814W.gst.matchfake',
                       '12257_NGC2213_F475W-F814W.gst.matchfake',
                       '9891_NGC1644_F555W-F814W.gst.matchfake',
                       '9891_NGC1795_F555W-F814W.gst.matchfake',
                       '9891_NGC1917_F555W-F814W.gst.matchfake']]
                       # '12257_HODGE6_F475W-F814W.gst.matchfake'
    asts_ = [asts.ASTs(fake) for fake in fakes]
    return asts_


def cmd_limits(targ, filter1):
    """
    zoom1_kw is the top right axis, HB
    zoom2_kw is the bottom right axis, MSTO
    xlim, ylim is the main axis.
    (ylim order doesn't matter, xlim order does)
    """
    default = {'F555W': {'xlim': [-0.5, 2.9],
                         'ylim': [26, 16],
                         'zoom1_kw': {'xlim': [0.8, 1.2],
                                      'ylim': [19.7, 18.7]},
                         'zoom2_kw': {'xlim': [0.15, 0.7],
                                      'ylim': [19.5, 21.5]}},
               'F475W': {'xlim': [-0.5, 2.9],
                         'ylim': [26, 16],
                         'zoom1_kw': {'xlim': [1.5, 1.9],
                                      'ylim': [20.7, 19.7]},
                         'zoom2_kw': {'xlim': [0.6, 1.1],
                                      'ylim': [22.23, 20.73]}}}

    kw = {'HODGE2': dict(default[filter1],
                         **{'zoom1_kw': {'xlim': [1.29, 1.6],
                                         'ylim': [19.5, 20.67]},
                            'zoom2_kw': {'xlim': [0.3, 0.74],
                                         'ylim': [19.35, 21.1]}}),
          'NGC1644': default[filter1],
          'NGC1718': default[filter1],
          'NGC1795': default[filter1],
          'NGC1917': default[filter1],
          'NGC1978': dict(default[filter1],
                          **{'zoom1_kw': {'xlim': [0.885, 1.25],
                                          'ylim': [19.7, 18.7]},
                             'zoom2_kw': {'xlim': [0.315, 0.795],
                                          'ylim': [19.9, 21.9]}}),
          'NGC2173': dict(default[filter1],
                          **{'zoom1_kw': {'xlim': [1.34, 1.63],
                                          'ylim': [19, 20.15]},
                             'zoom2_kw': {'xlim': [0.43, 0.86],
                                          'ylim': [20.1, 21.6]}}),
          'NGC2203': dict(default[filter1],
                          **{'zoom1_kw': {'xlim': [1.35, 1.62],
                                          'ylim': [19.3, 20.3]},
                             'zoom2_kw': {'xlim': [0.37, 0.85],
                                          'ylim': [19.87, 21.5]}}),
          'NGC2213': dict(default[filter1],
                          **{'zoom1_kw': {'xlim': [1.3, 1.62],
                                          'ylim': [20.2, 19.]},
                             'zoom2_kw': {'xlim': [0.4, 0.9],
                                          'ylim': [19.8, 21.5]}})}
    return kw[targ]


def cmd_plots(loc=None):
    """
    Produces two cmds
    one with full data set, one with the membership from asteca
    """
    here = os.getcwd()
    if loc is not None:
        os.chdir(loc)

    prefs = list({os.path.splitext(p)[0] for p in glob.glob('*memb*')})

    for pref in prefs:
        cluster = '{}.asteca'.format(pref.replace('_memb', ''))
        memb = '{}.dat'.format(pref)
        assert os.path.isfile(memb), '{0:s} not found'.format(memb)
        assert os.path.isfile(cluster), '{0:s} not found'.format(cluster)

        _, filters = parse_pipeline(memb)
        filter1, filter2 = filters
        targ = cluster.split('_')[1]

        cmd_kw = dict({'xy': False, 'zoom': True}, **cmd_limits(targ, filter1))
        fig, axs = cmd(cluster, filter1, filter2, **cmd_kw)
        fig, axs = cmd(memb, filter1, filter2, **cmd_kw, axs=axs, fig=fig,
                       plt_kw={'color': 'darkred'})
        axs[1].set_title('${}$'.format(targ))
        # if completeness:
        #     add_completeness(targ, comp=0.9, ax=axs[0])
        [ax.ticklabel_format(useOffset=False) for ax in axs]
        plt.savefig(os.path.join(here, '{}.pdf'.format(pref)))
        print('write {}'.format(os.path.join(here, '{}.pdf'.format(pref))))
        plt.close()

    os.chdir(here)


def make_fake_plots():
    """Produces one big ast.magdiff plot"""
    here = os.getcwd()
    sns.set_context('paper')

    asts_ = getfakes()

    fig, axss = plt.subplots(ncols=2, nrows=9, sharey=True, figsize=(4.5, 7.24))
    for i, ast in enumerate(asts_):
        ast.completeness(combined_filters=True, interpolate=True, binsize=0.15)
        ast.magdiff_plot(axs=axss[i])
        targ = ast.name.split('_')[1]
        axss[i][0].text(0.05, 0.05, '${}$'.format(targ),
                        transform=axss[i][0].transAxes)

    for ax in axss.ravel():
        ax.set_ylim(-2.02, 2.02)
        ax.set_xlim(14, 28)
        ax.tick_params(top='off', bottom='off', right='off', labelbottom='off')
        ax.grid()
        ax.set_ylabel(r'$\rm{In-Out}$')

    plt.yticks([-2, 0, 2])

    axbot = [axss[-1][0], axss[-1][1]]
    axtop = [axss[0][0], axss[0][1]]
    axright = axss.T[1]
    # turn back on ticks on bottom axes
    [ax.tick_params(bottom='on', labelbottom='on') for ax in axbot]
    # turn back on ticks on top axes
    [ax.tick_params(top='on', labeltop='on') for ax in axtop]

    # shift labels to the right on the right plots
    for ax in axright:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    # add 90% completeness lines
    for i, ast in enumerate(asts_):
        if i == 0:
            c1 = 25.
            c2 = 25.
        else:
            c1, c2 = ast.get_completeness_fraction(0.9)
        axss[i][0].axvline(c1, lw=2, color='darkred')
        axss[i][1].axvline(c2, lw=2, color='darkred')

    fig.subplots_adjust(bottom=0.1, top=0.95, hspace=0.35, wspace=0.07,
                        left=0.15, right=0.85)

    plt.savefig(os.path.join(here, 'fakes.pdf'))
    print('wrote {}'.format(os.path.join(here, 'fakes.pdf')))
    os.chdir(here)

if __name__ == "__main__":
    plt.style.use('presentation')
    sns.set_style('ticks')

    # cmd_plots()
    make_fake_plots()