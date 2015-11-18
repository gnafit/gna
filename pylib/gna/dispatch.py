import argparse
import sys
import os.path
from pkgutil import iter_modules
from gna.env import env
import gna.ui

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
    cls.initparser(subparsers.add_parser(name))
    opts = parser.parse_args(args)

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
        obj = cmdcls(cmdopts)
        obj.env = env.default
        obj.init()
        runlist.append(obj)

    for obj in runlist:
        obj.run()
