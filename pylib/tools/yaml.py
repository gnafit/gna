#!/usr/bin/env python

from __future__ import absolute_import
import yaml
from collections import OrderedDict, Iterable

def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """
    Ordered dict YAML loader
    from: https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    """
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    """Dump OrderedDict"""
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)

def yaml_load(string):
    """Parse multiple strings as yaml data. Multiple dictionaries are combined togather."""
    if isinstance(string, str):
        return ordered_load(string)

    if not isinstance(string, Iterable):
        raise TypeError('Invalid yaml_load argument type')

    ret = dict()
    for s in string:
        d=ordered_load(s, Loader=yaml.Loader)
        ret.update(d)
    return ret

def yaml_load_file(filename):
    """Load python dictionary from *.py file"""
    try:
        with open(filename, 'r') as stream:
            data = ordered_load(stream)
    except:
        raise Exception('Unable to load input data file: '+filename)

    return data

