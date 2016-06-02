import os

code = '/astro/store/phat3/phil/lmc_legacy/code/footprintfinder.py'
def callfoot(root_path):
    for root, _, filenames in os.walk(root_path):
        os.chdir(root)
        print('Working in directory: {}'.format(root))
	for i, filename in enumerate(filenames):
            if not filename.endswith('fits'):
                continue
            outf = filename.replace('.fits', '_footprint_ds9_linear.reg')
            if os.path.isfile(outf):
                print('{} exists. skipping...'.format(outf))
                continue
            print('Working on {} of {}'.format(i+1, len(filenames)))
            os.system('python {} -d {}'.format(code, filename))
	os.chdir(root_path)
    return
    
    
if __name__ == "__main__":
    callfoot(os.getcwd())    

