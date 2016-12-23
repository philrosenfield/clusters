import pandas as pd


def update_csv(fname, update_dict):
    new_name = '{}_extended.csv'.format(fname.replace('csv', ''))
    data = pd.read_csv(fname)
    for key, val in update_dict.items():
        data[key] = val
    data.to_csv(fname, index=False)
    print('wrote {}'.format(fname))
    return data


def combine_dataframes(dataframes, outfile='combined_files.csv'):
    all_data = pd.DataFrame()
    for dframe in dataframes:
        all_data = all_data.append(dframe, ignore_index=True)
    all_data.to_csv(outfile, index=False)
    print('wrote {}'.format(outfile))


update_sparcer_age = {'tmin': 9.34,
                      'tmax': 9.44,
                      'dt': 0.02,
                      'davv': 0.02,
                      'av0': 0.06,
                      'av1': 0.28,
                      'mu0': 18.44,
                      'mu1': 18.54,
                      'dmu': 0.02,
                      'dcol': 0.025,
                      'dmag': 0.05,
                      'dlogz': 0.015}

update_sparcer_z = {'tmin': 9.34,
                    'tmax': 9.44,
                    'dt': 0.02,
                    'davv': 0.02,
                    'av0': 0.06,
                    'av1': 0.28,
                    'mu0': 18.44,
                    'mu1': 18.54,
                    'dmu': 0.02,
                    'dcol': 0.025,
                    'dmag': 0.05,
                    'dlogz': 0.02}

update_finer_cmd = {'tmin': 9.34,
                    'tmax': 9.43,
                    'dt': 0.01,
                    'davv': 0.02,
                    'av0': 0.06,
                    'av1': 0.28,
                    'mu0': 18.44,
                    'mu1': 18.54,
                    'dmu': 0.02,
                    'dcol': 0.025,
                    'dmag': 0.05,
                    'dlogz': 0.015}


update_first_grid = {'tmin': 9.25,
                     'tmax': 9.65,
                     'dt': 0.01,
                     'davv': 0.05,
                     'av0': 0.05,
                     'av1': 0.25,
                     'mu0': 18.3,
                     'mu1': 18.6,
                     'dmu': 0.05,
                     'dcol': 0.05,
                     'dmag': 0.1,
                     'dlogz': 0.07}

update_dict = {'sparcer_age.csv': update_sparcer_age,
               'sparcer_z.csv', update_sparcer_z,
               'finer_cmd.csv', update_finer_cmd,
               'first_grid.csv', update_first_grid}


combine_dataframes([update_csv(fname, udict)
                    for fname, udict in update_dict.items()])
