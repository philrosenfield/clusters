"""Convert ASteCA membership file(s) to match photometry"""
from __future__ import print_function
import sys
import argparse
import numpy as np


def asteca2matchphot(filename):
    """Convert ASteCA membership file(s) to match photometry"""
    with open(filename, 'r') as inp:
        # asteca output header is #ID not # ID../
        header = inp.readline().strip().replace('#', '').split()
        data = np.genfromtxt(inp, names=header)

    phot = filename.replace('.dat', '.match')
    # this assumes V is the yfilter when running ASteCA.
    # i.e., mag2 = mag1 - color
    mag2 = data['mag'] - data['col1']

    np.savetxt(phot, np.column_stack([data['mag'], mag2]), fmt='%g')

    print('wrote {}'.format(phot))
    return phot


def main(argv):
    """
    Main function for converting ASteCA membership file(s) to match photometry
    """
    parser = argparse.ArgumentParser(
        description="Convert asteca membership file to match photmetery")

    parser.add_argument('inputfile', type=str, nargs='*',
                        help='asteca input file(s)')

    args = parser.parse_args(argv)

    _ = [asteca2matchphot(f) for f in args.inputfile]


if __name__ == "__main__":
    main(sys.argv[1:])
