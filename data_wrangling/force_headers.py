import argparse
import os
import sys

from .update_key import update_key

def force_headers(dry_run=True):
    """
    Walk through current directories and change all fits file headers
    """
    here = os.getcwd()
    for d in os.listdir('.'):
        if not '_' in d:
            continue
        if not os.path.isdir(d):
            continue
        pid, targ = d.split('_')
        if dry_run:
            print("{}:".format(d))
            print("update_key('PROPOSID', {}, fnames=fnames)".format(pid))
            print("update_key('TARGNAME', {}, fnames=fnames)".format(targ))
        else:
            os.chdir(d)
            fnames = [f for f in os.listdir('.') if f.endswith('fits')]
            update_key('PROPOSID', pid, fnames=fnames)
            update_key('TARGNAME', targ, fnames=fnames)
            os.chdir(here)
    return

def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        "update pid and targ in headers within a PID_TARG directory structure"))

    parser.add_argument("--dry_run", action="store_true",
                        help="print commands, don't execute")

    args = parser.parse_args(argv)

    force_headers(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

