#!/usr/bin/env python3
import pickle
import yaml
import numpy as np
import pprint
from tools.yaml import ordered_dump
from collections import OrderedDict

def main(opts):
    sens_min = None
    sens_max = None
    count = 0
    for datum in opts.files:
        if sens_min is None:
            try:
                exclude = datum['info']['exclude']
                if exclude==[]:
                    sens_min=datum['fun']
                count+=1
            except KeyError:
                pass

        if sens_max is None:
            try:
                include = datum['info']['include']
                if include==[]:
                    sens_max=datum['fun']
            except KeyError:
                pass

    print('Min, max, diff', sens_min, sens_max, sens_max-sens_min)

    data = OrderedDict()
    for datum in opts.files:
        try:
            lst = datum['info']['exclude']
            exclude=True
        except KeyError:
            lst = datum['info']['include']
            exclude=False
            pass

        if lst==[]:
            continue

        fun=datum['fun']
        pargroup = lst[0]
        if exclude:
            effect = fun - sens_min
            data.setdefault(pargroup, [0, 0])[1]=effect
        else:
            effect = sens_max - fun
            data.setdefault(pargroup, [0, 0])[0]=effect

    data=OrderedDict(
            fun_min      = sens_min,
            fun_max      = sens_max,
            sens         = data,
            names        = list(data.keys()),
            sens_include = list(v[0] for v in data.values()),
            sens_exclude = list(v[1] for v in data.values())
            )

    for name in opts.output:
        if name.endswith('.yaml'):
            with open(name, 'w') as f:
                ordered_dump(data, f, Dumper=yaml.Dumper)
        elif name.endswith('.pkl'):
            with open(name, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        else:
            raise Exception('Invalid output filename: '+name)
        print('Write output file:', name)


def load(fname):
    with open(fname, 'rb') as f:
        ret=pickle.load(f, encoding='latin1')['fitresult']['min']
        assert ret['success']
        return ret

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('files', nargs='+', type=load, help='input files')
    parser.add_argument('-o', '--output', nargs='+', default=[], help='output file to write')

    main( parser.parse_args() )
