from match.scripts.cmd import CMD
from match.scripts.likelihood import stellar_prob
from match.scripts.fileio import filename_data

cmd0 = CMD('9891_NGC1644_F555W-F814W.gst_bf0.95_imf0.7_tbin6e+07_vistep0.05_vstep0.1_dav0_ov0.50_ssp1774.out.cmd')
cmd0 = CMD('9891_NGC1644_F555W-F814W.gst_bf0.95_imf0.7_tbin6e+07_vistep0.05_vstep0.1_dav0_ov0.50_ssp1775.out.cmd')
cmd1 = CMD('9891_NGC1644_F555W-F814W.gst_bf0.95_imf0.75_tbin6e+07_vistep0.05_vstep0.1_dav0_ov0.50_ssp1776.out.cmd')
from match.scripts.graphics import match_plot
match_plot?
def compcmd(cmd1, cmd2):
   sig =
   hesslist = [cmd0.model, cmd1.model, cmd0.model-cmd1.model, cmd0.model-cmd1.model], cmd1.extent)
hesslist = [cmd0.model, cmd1.model, cmd0.model-cmd1.model, cmd0.model-cmd1.model], cmd1.extent]
hesslist = [cmd0.model, cmd1.model, cmd0.model-cmd1.model, cmd0.model-cmd1.model]]


def parse_args(argv=None):
   parser = argparse.ArgumentParser(
      description="Convert asteca membership file to match photmetery")

   parser.add_argument('cmd1', type=str, help='cmd file')
   parser.add_argument('cmd2', type=str, help='cmd file')

   return parser.parse_args(argv)

def main(argv=None):
   args = parse_args(argv=argv)
   cmd1 = CMD(args.cmd1)
   cmd2 = CMD(args.cmd2)
   compcmd(cmd1, cmd2)

_ = [asteca2matchphot(f) for f in args.inputfile]


if __name__ == "__main__":
   sys.exit(main())
