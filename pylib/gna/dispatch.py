from __future__ import print_function
import argparse
import sys
import os.path
from pkgutil import iter_modules
from gna.env import env
import gna.ui
from gna.config import cfg
from gna.packages import iterate_module_paths

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
    for pkgpath in cfg.pkgpaths: # TODO: deprecate
        modules.update({name: loader for loader, name, _ in iter_modules([pkgpath])})
    for pkgpath in iterate_module_paths('ui'):
        modules.update({name: loader for loader, name, _ in iter_modules([pkgpath])})
    return modules

def loadmodule(modules, name):
    loader = modules[name]
    return loader.find_module(name).load_module(name)

def loadcmdclass(modules, name, args):
    module=loadmodule(modules, name)
    cls = getattr(module, 'cmd')

    parserkwargs0 = getattr(cls, 'parserkwargs', {})
    parserkwargs = dict(dict(prog='gna -- {}'.format(name),
        description=cls.__doc__ or module.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter),
        **parserkwargs0)
    parser = argparse.ArgumentParser(**parserkwargs)
    cls.initparser(parser, env)
    opts = parser.parse_args(args, namespace=LazyNamespace())

    return cls, opts

def listmodules(modules, printdoc=False):
    print('Listing available modules. Module search paths:', ', '.join(cfg.pkgpaths))
    from textwrap import TextWrapper
    from os.path import relpath, isfile
    offsetlen, namelen = 2, 16
    offset  = ' '*offsetlen
    eoffset = '!'+offset[1:]
    docoffset = ' '*(offsetlen+namelen+6)
    wrp = TextWrapper(initial_indent=docoffset, subsequent_indent=docoffset)
    modnames = sorted(modules.
            keys())
    for modname in modnames:
        modname_print = modname.replace('_', '-')
        try:
            module=loadmodule(modules, modname)
        except Exception as e:
            print('{}{:<{namelen}s} from ... BROKEN: {}'.format(eoffset, modname_print, e.message, namelen=namelen))
        else:
            pyname = module.__file__
            if module.__file__.endswith('.pyc'):
                pyname = module.__file__[:-1]

            if isfile(pyname):
                modfile = './'+relpath(pyname)+' (pyc)'
                warning=''
            else:
                modfile = './'+relpath(module.__file__)
                warning='  \033[31mWarning!\033[0m The file {} does not exist. Consider removing all the \'*.pyc\' files from the project'.format(relpath(pyname))

            print('{}{:<{namelen}s} from {}{}'.format(offset, modname_print, modfile, warning, namelen=namelen))
            if printdoc and module.__doc__:
                for line in module.__doc__.strip().split('\n'):
                    print(wrp.fill(line))
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

        name = group[0].replace('-', '_')
        group = group[1:]
        if name not in modules:
            msg = 'unknown module %s' % name
            raise Exception(msg)
        cmdcls, cmdopts = loadcmdclass(modules, name, group)
        obj = cmdcls(env, cmdopts)
        obj.init()
        obj.run()
