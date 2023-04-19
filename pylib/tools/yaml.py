#!/usr/bin/env python

import yaml
from collections.abc import Iterable

def yaml_load(string):
    """Parse multiple strings as yaml data. Multiple dictionaries are combined togather."""
    if isinstance(string, str):
        return yaml.load(string, yaml.Loader)

    if not isinstance(string, Iterable):
        raise TypeError('Invalid yaml_load argument type')

    ret = dict()
    for s in string:
        d=yaml.load(s, yaml.Loader)
        ret.update(d)
    return ret

def yaml_load_file(filename):
    """Load python dictionary from *.py file"""
    try:
        with open(filename, 'r') as stream:
            data = yaml.load(stream, yaml.Loader)
    except:
        raise Exception('Unable to load input data file: '+filename)

    return data

