from __future__ import print_function
import argparse
import sys
import os.path
from pkgutil import iter_modules
from gna.env import env
import gna.ui
from gna.config import cfg

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
    modules = {}
    for pkgpath in cfg.pkgpaths:
        modules.update({name: loader for loader, name, _ in iter_modules([pkgpath])})
    return modules

def loadmodule(modules, name):
    loader = modules[name]
    return loader.find_module(name).load_module(name)

def loadcmdclass(modules, name, args):
    module=loadmodule(modules, name)
    cls = getattr(module, 'cmd')

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    cls.initparser(subparsers.add_parser(name), env)
    opts = parser.parse_args(args, namespace=LazyNamespace())

    return cls, opts

def listmodules(modules, printdoc=False):
    print('Listing available modules. Module search paths:', ', '.join(cfg.pkgpaths))
    from textwrap import TextWrapper
    from os.path import relpath
    offsetlen, namelen = 4, 20
    offset  = ' '*offsetlen
    eoffset = '!'+offset[1:]
    docoffset = ' '*(offsetlen+namelen+6)
    wrp = TextWrapper(initial_indent=docoffset, subsequent_indent=docoffset)
    for modname, module in modules.iteritems():
        try:
            module=loadmodule(modules, modname)
        except Exception as e:
            print('{}{:<{namelen}s} from ... BROKEN: {}'.format(eoffset, modname, e.message, namelen=namelen))
        else:
            print('{}{:<{namelen}s} from {}'.format(offset, modname, './'+relpath(module.__file__), namelen=namelen))
            if printdoc and module.__doc__:
                print(wrp.fill(module.__doc__))
                print()

def run():
    modules = getmodules()
    for group in arggroups(sys.argv):
        if not group:
            continue
        if group==['--list']:
            listmodules(modules)
            sys.exit(0)
        elif group==['--list-long']:
            listmodules(modules, True)
            sys.exit(0)

        name = group[0] = group[0].replace('-', '_')
        if name not in modules:
            msg = 'unknown module %s' % name
            raise Exception(msg)
        cmdcls, cmdopts = loadcmdclass(modules, name, group)
        obj = cmdcls(env, cmdopts)
        obj.init()
        obj.run()
