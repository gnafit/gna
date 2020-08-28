"""Current working dir for GNA"""

from os.path import join, exists, isdir
from os import makedirs, access, W_OK
from collections import Iterable

_cwd = ''

def get_cwd():
    """Return CWD"""
    return _cwd

def set_cwd(cwd):
    """Set CWD
    Create if needed
    """
    global _cwd
    _cwd = cwd
    _make_cwd()

def get_path(path):
    """Return the path. Prepend with CWD"""
    if not _cwd:
        return path

    return join(_cwd, path)

def update_namespace_cwd(ns, keys):
    """Update Argparser namespace by prepending all the paths with CWD"""
    if not _cwd:
        return

    if isinstance(keys, str):
        keys = keys,

    for key in keys:
        val=ns.__dict__[key]

        if val is None:
            continue

        if isinstance(val, str):
            ns.__dict__[key] = join(_cwd, val)
        elif isinstance(val, Iterable):
            ns.__dict__[key] = map(lambda s: join(_cwd, s), val)
        else:
            raise Exception('Unexpected value {!s} type: {}. Should be string or list of strings.'.format(val, type(val).__name__))

def _make_cwd():
    if exists(_cwd):
        if not isdir(_cwd):
            raise Exception('Unable to use {} as CWD: already exists, not a directory'.format(_cwd))
    else:
        print('Create CWD:', _cwd)
        makedirs(_cwd)

    if not access(_cwd, W_OK):
        raise Exception('CWD {} is not writable'.format(_cwd))
