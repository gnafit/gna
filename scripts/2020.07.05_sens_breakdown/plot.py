#!/usr/bin/env python3

from matplotlib import pyplot as plt
import pickle
import numpy as np
from tabulate import tabulate
from mpl_tools.helpers import savefig

changenames = {
        'lsnl_weight': 'LSNL',
        'rate_norm': 'Bkg rate',
        'bkgshape': 'Bkg shape',
        'SinSqDouble13': r'$\sin^22\theta_{13}$',
        'juno.norm': 'Normalization',
        'eres': 'Energy resolution',
        'offeq_scale': 'Off-equilibrium',
        'snf_scale': 'SNF',
        'thermal_power_scale': 'Thermal power',
        'fission_fractions_scale': 'Fission fractions',
        'energy_per_fission_scale': 'Energy/fission',
        }

namestext = {
        r'$\sin^22\theta_{13}$': u'sin²2θ₁₃'
        }

def combine_data(names, data, threshold):
    for i in range(1, data.size):
        if data[:i].sum()>threshold:
            break
    if i==1:
        return names, data
    i-=1
    comb = data[:i].sum()

    names = ['Other']+names[i:]
    data = np.concatenate(([comb],data[i:]), axis=None)
    return names, data

def plot_pie(names, data, title):
    isort = np.argsort(data)
    names = names[isort].tolist()
    data = data[isort]

    print(title)
    print('Data:', data)
    threshold = 0.03
    names, data=combine_data(names, data, threshold)
    print('Data combined ({}):'.format(threshold), data)

    data=data[::-1]
    names=names[::-1]

    fig = plt.figure()
    ax = plt.subplot(111, title=title)

    labels = [ '{}: {:.2f}%'.format(n, d*100) for n, d in zip(names, data) ]
    ax.pie(data, labels=labels)

def main(opts):
    fun_min = opts.input['fun_min']
    fun_max = opts.input['fun_max']
    fun_diff = fun_max - fun_min
    names = np.array([changenames.get(name, name) for name in opts.input['names']])

    include = np.array(opts.input['sens_include'])
    include_total = include.sum()
    include_overshoot = include_total - fun_diff

    exclude = np.array(opts.input['sens_exclude'])
    exclude_total = exclude.sum()
    exclude_overshoot = exclude_total - fun_diff

    print('Min, max, diff', fun_min, fun_max, fun_diff)
    print('Include total, overshoot', include_total, include_overshoot)
    print('Exclude total, overshoot', exclude_total, exclude_overshoot)

    plot_pie(names, exclude/exclude.sum(), 'Full-syst')
    savefig(opts.output, suffix='_exclude')
    plot_pie(names, include/include.sum(), 'None+syst')
    savefig(opts.output, suffix='_include')

    comb_sum = (exclude+include)/(exclude.sum()+include.sum())
    comb_avg = (exclude/exclude.sum()+include/include.sum())
    comb_avg = comb_avg/comb_avg.sum()

    plot_pie(names, comb_sum, 'Combine (sum)')
    savefig(opts.output, suffix='_comb_sum')
    plot_pie(names, comb_avg, 'Combine (avg)')
    savefig(opts.output, suffix='_comb_avg')

    header = [ 'Name', 'None+syst', '%', 'Full-syst', '%', 'Combine (sum), % ▼', 'Combine (avg), %' ]
    table = []
    for i in range(len(names)):
        table.append( [namestext.get(names[i],names[i]),
            include[i], 100.0*include[i]/include.sum(),
            exclude[i], 100.0*exclude[i]/exclude.sum(),
            100.0*comb_sum[i], 100.0*comb_avg[i]
        ] )
    table = sorted(table, key=lambda a: a[5], reverse=True)
    floatfmt = ('.2f', '.5f', '.2f', '.5f', '.2f', '.2f', '.2f')
    t = tabulate(table, header, floatfmt=floatfmt)
    print(t)

    if opts.latex:
        with open(opts.latex, 'w') as f:
            t = tabulate(table, header, floatfmt=floatfmt, tablefmt='latex_booktabs')
            f.write(t)
            print('Write output file:', opts.latex)

    plt.show()

def load(fname):
    with open(fname, 'rb') as f:
        ret=pickle.load(f)
        return ret

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('input', type=load)
    parser.add_argument('-o', '--output', help='output file')
    parser.add_argument('-l', '--latex', help='output table file')

    main(parser.parse_args())
