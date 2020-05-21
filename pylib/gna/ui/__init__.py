# -*- coding: utf-8 -*-
from __future__ import absolute_import
import argparse
from gna.env import PartNotFoundError

def _expandprefix(prefix, allowed):
    matches = [k for k in allowed if k.startswith(prefix)]
    if len(matches) > 1:
        msg = "ambigous prefix {0!r}, candidates: {1}"
        raise Exception(msg.format(prefix, ', '.join(matches)))
    elif not matches:
        msg = "invalid prefix {0!r}, possible full values: {1}"
        raise Exception(msg.format(prefix, ', '.join(allowed)))
    else:
        return matches[0]

class at_least(object):
    def __init__(self, count, f):
        self.count = count
        self.f = f

class qualified(object):
    def __init__(self, *args):
        self.types = args

class lazyproperty(object):
    def __init__(self, f):
        self.f = f

def append_typed(*types, **kwargs):
    lazy = kwargs.get('lazy', False)
    lazyitems = []
    class AppendTypedAction(argparse._AppendAction):
        def __call__(self, parser, namespace, values, option_string=None):
            def gettyped(f, v):
                try:
                    return f(v)
                except PartNotFoundError as e:
                    msg = "no {0} named {1!r}"
                    raise argparse.ArgumentError(self, msg.format(e.parttype, e.partname))

            def resolvetypes(fs, vs):
                res = [gettyped(f, v) for f, v in zip(fs, vs)]
                if self.nargs is None:
                    return res[0]
                else:
                    return res

            funcs = []
            newvalues = []
            if self.nargs is None:
                values = [values]
            for i, (f, v) in enumerate(zip(types, values)):
                if isinstance(f, at_least):
                    rest = values[i:]
                    if len(rest) < f.count:
                        msg = 'expected at least {0} arguments'
                        raise argparse.ArgumentError(self, msg.format(i+f.count))
                    funcs.append(lambda x: list(map(f.f, x)))
                    newvalues.append(rest)
                    break
                elif isinstance(f, qualified):
                    try:
                        prefix, name = v.split(":", 1)
                    except ValueError:
                        msg = 'qualified name expected, got {0}'
                        raise argparse.ArgumentError(self, msg.format(v))
                    allowed = [getattr(t, 'parttype', None) or t.__name__ for t in f.types]
                    prefix = _expandprefix(prefix, allowed)
                    funcs.append(f.types[allowed.index(prefix)])
                    newvalues.append(name)
                else:
                    funcs.append(f)
                    newvalues.append(v)
            else:
                if len(newvalues) < len(values):
                    msg = 'expected at most {0} arguments'
                    raise argparse.ArgumentError(self, msg.format(len(types)))
                elif len(newvalues) > len(values):
                    msg = 'expected at least {0} arguments'
                    raise argparse.ArgumentError(self, msg.format(len(types)))
            if not lazy:
                items = getattr(namespace, self.dest, [])
                items.append(resolvetypes(funcs, newvalues))
                setattr(namespace, self.dest, items)
            else:
                if not lazyitems:
                    prop = lazyproperty(lambda: [resolvetypes(*x) for x in lazyitems])
                    setattr(namespace, self.dest, prop)
                lazyitems.append((funcs, newvalues))

    return AppendTypedAction

def set_typed(parttype):
    class SetTypedAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            try:
                setattr(namespace, self.dest, parttype(values))
            except PartNotFoundError as e:
                msg = "no {0} named {1!r}"
                raise argparse.ArgumentError(self, msg.format(e.parttype, values))

    return SetTypedAction

class basecmd(object):
    @classmethod
    def initparser(cls, parser, env):
        pass

    def __init__(self, env, opts):
        self.env = env
        self.opts = opts

    def init(self):
        pass

    def run(self):
        pass
