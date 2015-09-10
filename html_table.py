import argparse
import os
import subprocess
import sys

header = '<!DOCTYPE html>\n<html lang="en-US">\n<title>{}</title>\n<body>\n'
footer = '</body>\n</html>\n'

def w_h_frompng(img):
    '''find the width had height of a png.'''
    info = subprocess.Popen([r"file {}".format(img)], shell=True,
                            stdout=subprocess.PIPE).communicate()[0]
    w, h = info.split(',')[1].translate(None, ' ').split('x')
    return w, h

def one_col(imgpath, webpath, defaults={}):
    '''
    create a string that has the imgpath as title and links to the image
    on the webpath.
    '''
    d = {}
    fmt = '<h3>{imgname}</h3>\n<p><a href=\'{imgloc}\'><img height=\'{imgh}\' src=\'{imgloc}\' width=\'{imgw}\' alt=\'{imgname}\'></a></p>\n'
    d['imgname'] = os.path.split(imgpath)[1]
    d['imgloc'] = os.path.join(webpath, d['imgname'])
    d['imgw'], d['imgh'] = w_h_frompng(imgpath)
    # overwrite from defaults (e.g., image scaling)
    d = dict(d.items() + defaults.items())
    return fmt.format(**d)


def write_script(args):
    """
    Write a script to tar and scp files to a server.
    """
    imgs = ' '.join(args.images)
    line = 'tar -cvf imgs.tar {} {}\ngzip imgs.tar\n'.format(args.outfile, imgs)
    script = args.outfile + '.sh'
    with open(script, 'w') as outp:
        outp.write(line)
    print('wrote {}'.format(script))

def main(argv):
    parser = argparse.ArgumentParser(description="make a one column html file of images")

    parser.add_argument('-o', '--outfile', type=str, default='imgs.html',
                        help='output html to write to')

    parser.add_argument('-s', '--script', action='store_true',
                        help='write a push to server script')

    parser.add_argument('-f', '--clobber', action='store_true',
                        help='write a new file if one exists (append by default)')

    parser.add_argument('-p', '--path', default='mc_legacy/', help='path from webbase')

    parser.add_argument('-w', '--webbase', type=str, default='https://www.cfa.harvard.edu/~prosenfi/',
                        help='image http address')

    parser.add_argument('images', type=str, nargs='*',
                        help='image(s) to put into html')

    args = parser.parse_args(argv)
    args.webbase = os.path.join(args.webbase, args.path)

    # not really an append...
    if args.clobber or not os.path.isfile(args.outfile):
        # write the header if this will be a new file
        line = header.format(args.outfile.split('.')[0])
    else:
        # erase the footer ... could be more general.
        with open(args.outfile, 'r') as inp:
            line = ''.join(inp.readlines()[:-2])

    line += ''.join([one_col(img, args.webbase) for img in args.images])
    line += footer

    with open(args.outfile, 'w') as outp:
        outp.write(line)
    print('wrote to {}'.format(args.outfile))

    if args.script:
        write_script(args)

if __name__ == "__main__":
    main(sys.argv[1:])
