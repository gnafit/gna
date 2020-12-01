#!/usr/bin/env python

from argparse import Action, _copy, _ensure_value, ArgumentParser

def AppendSubparser( *args, **kwargs ):
    class AppendSubparserClass(Action):
        subparser = ArgumentParser( *args, **kwargs )
        def __init__(self, option_strings, dest, nargs=None, const=None, default=None,
                     type=None, choices=None, required=False, help=None, metavar=None):
            if nargs == 0:
                raise ValueError('nargs for append actions must be > 0; if arg '
                                'strings are not supplying the value to append, '
                                'the append const action may be more appropriate')
            if const is not None and nargs != OPTIONAL:
                raise ValueError('nargs must be %r to supply const' % OPTIONAL)
            super(AppendSubparserClass, self).__init__(
                option_strings=option_strings, dest=dest, nargs=nargs, const=const,
                default=default, type=type, choices=choices, required=required,
                help=help, metavar=metavar)

        def __call__(self, parser, namespace, values, option_string=None):
            items = _copy.copy(_ensure_value(namespace, self.dest, []))
            items.append( self.subparser.parse_args(values) )
            setattr(namespace, self.dest, items)

        @classmethod
        def add_argument(cls, *args, **kwargs):
            return cls.subparser.add_argument( *args, **kwargs )

        @classmethod
        def add_mutually_exclusive_group(cls, *args, **kwargs):
            return cls.subparser.add_mutually_exclusive_group( *args, **kwargs )

    return AppendSubparserClass

def SingleSubparser( *args, **kwargs ):
    class SingleSubparserClass(Action):
        subparser = ArgumentParser( *args, **kwargs )
        def __init__(self, option_strings, dest, nargs=None, const=None, default=None,
                     type=None, choices=None, required=False, help=None, metavar=None):
            if nargs == 0:
                raise ValueError('nargs for append actions must be > 0; if arg '
                                'strings are not supplying the value to append, '
                                'the append const action may be more appropriate')
            if const is not None and nargs != OPTIONAL:
                raise ValueError('nargs must be %r to supply const' % OPTIONAL)
            super(SingleSubparserClass, self).__init__(
                option_strings=option_strings, dest=dest, nargs=nargs, const=const,
                default=default, type=type, choices=choices, required=required,
                help=help, metavar=metavar)

        def __call__(self, parser, namespace, values, option_string=None):
            items=self.subparser.parse_args(values)
            setattr(namespace, self.dest, items)

        @classmethod
        def add_argument(cls, *args, **kwargs):
            return cls.subparser.add_argument( *args, **kwargs )

        @classmethod
        def add_mutually_exclusive_group(cls, *args, **kwargs):
            return cls.subparser.add_mutually_exclusive_group( *args, **kwargs )

    return SingleSubparserClass
