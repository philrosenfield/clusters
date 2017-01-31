"""Write out a the unique RAs of a file"""
import argparse
import pandas as pd
import numpy as np
import sys

def uniqify(fin, column='s_ra'):
    fext = '_unique_{}.csv'.format(column)
    fout = fin.replace('.csv', fext)

    data = pd.read_csv(fin, header=0)
    if 'ra' in column and not column in data.columns:
        column = 'RA (J2000)'
        assert columnn in data.columns, \
            'column not found {}'.format(data.columns)

    _, inds = np.unique(data[column], return_index=True)

    df = data.loc[inds]
    df.to_csv(fout, index=False)
    print('wrote {0:s}'.format(fout))
    return


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="uniqify a file based on column")

    parser.add_argument('-c', '--column', type=str, default='s_ra',
                        help='specify column')
    parser.add_argument('file', type=str, help='csv file to uniqify')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    uniqify(args.file, column=args.column)


if __name__ == "__main__":
    sys.exit(main())
