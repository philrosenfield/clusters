import os
import seaborn as sns
import matplotlib.pyplot as plt
from match.scripts.utils import parse_pipeline
from match.scripts import asts
from clusters.data_plots import cmd


def getfakes():
    loc = '/Users/rosenfield/research/clusters/asteca/acs_wfc3/match_runs'
    fakes = [os.path.join(loc, f)
             for f in ['9891_NGC1978_F555W-F814W.gst.matchfake',
                       '12257_NGC1718_F475W-F814W.gst.matchfake',
                       '12257_NGC2173_F475W-F814W.gst.matchfake',
                       '12257_NGC2203_F475W-F814W.gst.matchfake',
                       '12257_NGC2213_F475W-F814W.gst.matchfake',
                       '9891_NGC1917_F555W-F814W.gst.matchfake',
                       '9891_NGC1795_F555W-F814W.gst.matchfake',
                       '9891_NGC1644_F555W-F814W.gst.matchfake',
                       '12257_HODGE6_F475W-F814W.gst.matchfake',
                       '12257_HODGE2_F475W-F814W.gst.matchfake']]
    asts_ = [asts.ASTs(fake) for fake in fakes]
    return asts_


def cmd_plots():
    """
    Produces two cmds
    one with full data set, one with the membership from asteca
    """
    here = os.getcwd()
    os.chdir('/Users/rosenfield/research/clusters/asteca/acs_wfc3/')
    prefs = ['12257_HODGE2_F475W-F814W_uvis.gst',
             '9891_NGC1644_F555W-F814W.gst',
             '9891_NGC1795_F555W-F814W.gst',
             '9891_NGC1917_F555W-F814W.gst',
             '12257_NGC2203_F475W-F814W_uvis.gst',
             '12257_NGC2213_F475W-F814W_uvis.gst',
             '12257_NGC1718_F475W-F814W_uvis.gst',
             '12257_NGC2173_F475W-F814W_uvis.gst',
             '12257_HODGE6_F475W-F814W_uvis.gst',
             '9891_NGC1978_F555W-F814W.gst']
    sns.set_context('paper', font_scale=3)
    for pref in prefs:
        cluster = '{}.asteca'.format(pref)
        memb = '{}_memb.dat'.format(pref)
        _, filters = parse_pipeline(memb)
        filter1, filter2 = filters
        targ = cluster.split('_')[1]
        print(targ)
        fig, axs = cmd(cluster, filter1, filter2, xy=False)
        fig, axs = cmd(memb, filter1, filter2, axs=axs[0], fig=fig,
                       plt_kw={'color': 'darkred'}, xy=False)
        if completeness:
            add_completeness(targ, comp=0.9, ax=axs[0])
        [ax.ticklabel_format(useOffset=False) for ax in axs]
        # [ax.grid() for ax in axs]
        if len(axs) > 1:
            axs[1].yaxis.set_label_position("right")
            axs[1].yaxis.tick_right()
            fig.subplots_adjust(wspace=0.1, left=0.07, right=0.9, top=0.95)
        axs[0].text(0.8, 0.95, '${}$'.format(targ), fontsize=20,
                    transform=axs[0].transAxes)
        plt.savefig(os.path.join(here, '{}.pdf'.format(pref)))
        print('write {}'.format(os.path.join(here, '{}.pdf'.format(pref))))
        plt.close()

    os.chdir(here)


def make_fake_plots():
    """Produces one big ast.magdiff plot"""
    here = os.getcwd()
    sns.set_context('paper')

    asts_ = get_fakes()

    fig, axss = plt.subplots(ncols=2, nrows=10, sharey=True,
                             figsize=(4.5, 7.24))
    for i, ast in enumerate(asts_):
        ast.completeness(combined_filters=True, interpolate=True, binsize=0.15)
        ast.magdiff_plot(axs=axss[i])
        targ = fakes[i].split('_')[1]
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
        c1, c2 = ast.get_completeness_fraction(0.9)
        axss[i][0].axvline(c1, lw=2, color='darkred')
        axss[i][1].axvline(c2, lw=2, color='darkred')

    fig.subplots_adjust(bottom=0.1, top=0.95, hspace=0.35, wspace=0.07,
                        left=0.11, right=0.90)

    plt.savefig(os.path.join(here, 'fakes.pdf'))
    print('wrote {}'.format(os.path.join(here, 'fakes.pdf')))
    os.chdir(here)

if __name__ == "__main__":
    plt.style.use('presentation')
    sns.set_style('ticks')

    cmd_plots()
    make_fake_plots()
