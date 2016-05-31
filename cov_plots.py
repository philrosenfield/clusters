data = pd.read_table('all_eeps.csv', delim_whitespace=True, header=0)
lss = ['-', '--', '-.', ':']
colors = ['k', (0.8, 0.7254901960784313, 0.4549019607843137),
          (0.39215686274509803, 0.7098039215686275, 0.803921568627451),
          (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
          (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
          (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
          (0.5058823529411764, 0.4470588235294118, 0.6980392156862745)]
fig, ax = plt.subplots()
izahb = data['iptcri'] == 11
ishb = data['hb'] == 1
zahb = izahb & ishb
for i, z in enumerate(np.unique(data['Z'])):
    for j, ov in enumerate(np.unique(data['OV'])):
        lab = '{} {}'.format(z, ov)
        iov = data['OV'] == ov
        df = data[zahb & iov]
        xcol = df['logT']
        ycol = df['logL']
        xcol = df['F555W'] - df['F814W']
        ycol = df['F814W']
        ax.plot(xcol, ycol, label=lab, color=colors[i], ls=lss[j])
        #[ax.text(xcol.iloc[k], ycol.iloc[k], df['mass'].iloc[k]) for k in range(len(df['logT']))]

ax.set_xlim(ax.get_xlim()[::-1])


ov

lss = ['-', '--', '-.', ':']
colors = ['k', (0.8, 0.7254901960784313, 0.4549019607843137),
          (0.39215686274509803, 0.7098039215686275, 0.803921568627451),
          (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
          (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
          (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
          (0.5058823529411764, 0.4470588235294118, 0.6980392156862745)]
fig, ax = plt.subplots()
izahb = data['iptcri'] == 11
ishb = data['hb'] == 1
zahb = izahb & ishb
for i, z in enumerate(np.unique(data['Z'])[:-1]):
    iz = data['Z'] == z
        df3 = data[zahb & (data['OV'] == 0.3)]
        df5 = data[(data['iptcri'] == 11) & (data['Z'] == z) & (data['OV'] == 0.5) & (data['hb'] == 1)]
        df6 = data[(data['iptcri'] == 11) & (data['Z'] == z) & (data['OV'] == 0.6) & (data['hb'] == 1)]
        ax.plot(map(float, df5['logT']), map(float, df5['logL']), label=lab,
                color=colors[i])
        ax.plot(map(float, df5['logT']), map(float, df5['logL']), label=lab,
                color=colors[i])

        #[ax.text(float(df['logT'].iloc[k]), float(df['logL'].iloc[k]), df['mass'].iloc[k]) for k in range(len(df['logT']))]

ax.set_xlim(ax.get_xlim()[::-1])
