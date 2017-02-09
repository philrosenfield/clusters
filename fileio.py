"""Methods to read and write files"""
from __future__ import print_function, absolute_import
import os
import glob
import sys


def get_files(src, search_string):
    '''
    returns a list of files, similar to ls src/search_string
    '''
    if not src.endswith('/'):
        src += '/'
    try:
        files = glob.glob1(src, search_string)
    except IndexError:
        logger.error('Can''t find %s in %s' % (search_string, src))
        sys.exit(2)
    files = [os.path.join(src, f)
             for f in files if os.path.isfile(os.path.join(src, f))]
    return files


def get_dirs(src, criteria=None):
    """
    return a list of directories in src, optional simple cut by criteria

    Parameters
    ----------
    src : str
        abs path of directory to search in
    criteria : str
        simple if criteria in d to select within directories in src

    Returns
    -------
    dirs : abs path of directories found
    """
    dirs = [os.path.join(src, l) for l in os.listdir(src) if os.path.join(src, l)]
    if criteria is not None:
        dirs = [d for d in dirs if criteria in d]
    return dirs
