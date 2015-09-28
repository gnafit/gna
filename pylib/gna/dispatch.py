import argparse
import sys
import os
import os.path
from pkgutil import iter_modules
from gna.env import env

def arggroups(argv):
    breaks = [i+1 for i, arg in enumerate(argv[1:], 1) if arg == '--']
    for start, end in zip([1]+breaks, [i-1 for i in breaks]+[len(argv)]):
        yield argv[start:end]

def getmodules():
    allmodules = {}
    basedir = os.path.join(os.path.dirname(__file__), "modules")
    for subdir in os.listdir(basedir):
        dpath = os.path.join(basedir, subdir)
        if not os.path.isdir(dpath):
            continue
        if os.path.exists(os.path.join(dpath, "__init__.py")):
            continue
        modules = {name: loader for loader, name, _ in iter_modules([dpath])}
        conflicts = set(modules.keys()).intersection(allmodules.keys())
        if conflicts:
            msg = "conflicting modules: {0}".format(', '.join(conflicts))
            raise Exception(msg)
        allmodules.update(modules)
    return allmodules

def loadcmdclass(modules, name, args):
    loader = modules[name]
    module = loader.find_module(name).load_module(name)
    cls = getattr(module, name)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    cls.initparser(subparsers.add_parser(cls.__name__))
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
        obj.env = env
        obj.init()
        runlist.append(obj)

    for obj in runlist:
        obj.run()
