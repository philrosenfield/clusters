import os
import numpy as np


def check_dirs(f):
    '''
    will make all dirs necessary for input to be an existing directory.
    if input does not end with '/' it will add it, and then make a directory.
    '''
    if not f.endswith('/'):
        f += '/'

    d = os.path.dirname(f)
    if not os.path.isdir(d):
        os.makedirs(d)
        print('made dirs: {}'.format(d))

def move_files(inparr, dest_dir, root):
    for i in inparr:
        os.system('mv {}/{} {}'.format(root, i, dest_dir))

def only_flc(root_path):
    def check_if_flc(inparr, flcarr, extstr, subdir):
        fname = [f.split('_')[0] for f in flcarr]
        iname = [f.split('_')[0] for f in inparr]
        imatches = np.concatenate([[i for i, n in enumerate(iname) if n == f]
                                    for f in fname])
        imatches = np.array(imatches, dtype=int)
        if len(imatches) == 0:
            print('no overlapping {} with flc in {}'.format(extstr, subdir))
        return np.asarray(inparr)[imatches]

    def flc_dups(inparr, flcarr, extstr, root):
        if len(inparr) > 0:
            dups = check_if_flc(inparr, flcarr, extstr, subdir)
            if len(dups) > 0:
                move_files(dups, mvtodir, root)
        else:
            print('no {} in {}'.format(extstr, subdir))
        return

    root, diry = os.path.split(root_path)
    new_path = os.path.join(root, 'dup_ext', diry)

    for root, _, filenames in os.walk(root_path):
        flcs = [f for f in filenames if 'flc' in f]
        raws = [f for f in filenames if 'raw' in f]
        flts = [f for f in filenames if 'flt' in f]
        subdir = os.path.split(root)[1]
        
        if len(flcs) == 0:
            print('no flcs in {}'.format(subdir))
            continue
        
        mvtodir = os.path.join(new_path, subdir)
        check_dirs(mvtodir)
        flc_dups(raws, flcs, 'raw', root)
        flc_dups(flts, flcs, 'flt', root)
    return
    
    
if __name__ == "__main__":
    only_flc(os.getcwd())    

