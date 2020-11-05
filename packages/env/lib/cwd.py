"""Current working dir for GNA"""

from os.path import join, exists, isdir
from os import makedirs, access, W_OK
from collections import Iterable

_cwd = ''
_prefix = ''

_processed_paths = []

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

def get_prefix():
    """Get prefix"""
    return _prefix

def set_prefix(prefix):
    """Set prefix"""
    global _prefix
    _prefix=prefix

def storepaths(fcn):
    global _processed_paths
    def newfcn(path):
        ret = fcn(path)
        _processed_paths.append(ret)
        return ret
    return newfcn

def get_processed_paths():
    return _processed_paths

@storepaths
def get_path(path):
    """Return the path. Prepend with CWD and prefix
    NOTE: prefix is applied only if path does not contain /
    """
    if _prefix and not '/' in path:
        path = _prefix + path

    if not _cwd:
        return path

    return join(_cwd, path)

def update_namespace_cwd(ns, keys):
    """Update Argparser namespace by prepending all the paths with CWD"""
    if not _cwd and not _prefix:
        return

    if isinstance(keys, str):
        keys = keys,

    for key in keys:
        val=ns.__dict__[key]

        if val is None:
            continue

        if isinstance(val, str):
            ns.__dict__[key] = get_path(val)
        elif isinstance(val, Iterable):
            ns.__dict__[key] = map(get_path, val)
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
