# -*- coding: utf-8 -*-
#! /usr/bin/env python3
import os
import argparse
import warnings
from itertools import chain
import glob
from collections import defaultdict
import numpy as np
import h5py

def _check_pathes(pathes, strict=True):
    filtered = []
    for path in pathes:
        if h5py.is_hdf5(path):
            filtered.append(path)
        else:
            if strict:
                raise ValueError("{} is not an HDF5 file".format(path))
            else:
                warnings.warn("Dropping {} since it isn't HDF5 file".format(path))
    return filtered

class Glob_HDF5(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        pathes = []
        for path in values:
            pathes.extend(glob.glob(path))

        if not len(pathes):
            raise ValueError("The glob expansion result is empty")
        filtered = _check_pathes(pathes, strict=False)
        final = list(set(filtered))
        setattr(namespace, self.dest, final)

def get_all_keys(obj, keys=None):
    if keys == None:
        keys=[]
    keys.append(obj.name)
    if isinstance(obj, h5py.Group):
        for item in obj:
            if isinstance(obj[item], h5py.Group):
                get_all_keys(obj[item], keys)
            else: # isinstance(obj[item], h5py.Dataset):
                keys.append(obj[item].name)
    return keys

def _depth(hdf_key):
    return hdf_key.count('/')

class Merger():
    '''Simple CLI utility to merge HDF5 files with chi-square maps
    after likelihood profiling in different segments of the grid
    produced by scan module. Top-level attributes are assumed to be identical for all
    merged files'''
    def __init__(self, opts):
        with h5py.File(opts.output, 'w') as f:
            for path in opts.input:
                input_file = h5py.File(path, 'r')
                for key in input_file:
                    try:
                        # easy case: recursively copy entire group
                        input_file.copy(key, f)
                    except ValueError as e:
                        # hard case: the group got splitted between files and
                        # simply copying won't work, need to identify what
                        # groups already in the output and update it and
                        # then copy others
                        keys_in_input = set(get_all_keys(input_file[key]))
                        keys_in_ouput = set(get_all_keys(f[key]))
                        missing_keys = list(keys_in_input.difference(keys_in_ouput))

                        # sort keys so groups come before datasets
                        missing_keys.sort(key=_depth)

                        # make sure each missing group is created, attributes
                        # and datasets are copied
                        for missed_key in missing_keys:
                            input_object = input_file[missed_key]
                            if isinstance(input_object, h5py.Group):
                                f.require_group(missed_key)
                                for name, val in input_object.attrs.items():
                                    f[missed_key].attrs.create(name, val)
                            if isinstance(input_object, h5py.Dataset):
                                f.create_dataset(missed_key, data=input_object[:])

                for attr, value in input_file['/'].attrs.items():
                    f['/'].attrs[attr] = value

                input_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Simple tool to merge HDF5 after
             likelihood profiling by scan module''')
    parser.add_argument('input', nargs='*', type=os.path.abspath,
           action=Glob_HDF5, help='List of HDF5 files to merge with possible globbing, duplicates are removed')
    parser.add_argument('--output', type=os.path.abspath, required=True,
            help='Path to merged output file')
    opts = parser.parse_args()
    merger = Merger(opts)
