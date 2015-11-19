import argparse
import sys
import os.path
from pkgutil import iter_modules
from gna.env import env
import gna.ui

class LazyNamespace(argparse.Namespace):
    def __getattribute__(self, name):
        attr = super(LazyNamespace, self).__getattribute__(name)
        try:
            call = isinstance(attr, gna.ui.lazyproperty)
        except TypeError:
            call = False
        if call:
            attr = attr.f()
            setattr(self, name, attr)
        return attr

def arggroups(argv):
    breaks = [i+1 for i, arg in enumerate(argv[1:], 1) if arg == '--']
    for start, end in zip([1]+breaks, [i-1 for i in breaks]+[len(argv)]):
        yield argv[start:end]

def getmodules():
    pkgpath = os.path.dirname(gna.ui.__file__)
    modules = {name: loader for loader, name, _ in iter_modules([pkgpath])}
    return modules

def loadcmdclass(modules, name, args):
    loader = modules[name]
    module = loader.find_module(name).load_module(name)
    cls = getattr(module, 'cmd')

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    cls.initparser(subparsers.add_parser(name), env.default)
    opts = parser.parse_args(args, namespace=LazyNamespace())

    return cls, opts

def run():
    modules = getmodules()
    runlist = []
    for group in arggroups(sys.argv):
        if not group:
            continue
        name = group[0]
        if name not in modules:
            msg = 'unknown module %s' % name
            raise Exception(msg)
        cmdcls, cmdopts = loadcmdclass(modules, name, group)
        obj = cmdcls(env.default, cmdopts)
        obj.init()
        runlist.append(obj)

    for obj in runlist:
        obj.run()
