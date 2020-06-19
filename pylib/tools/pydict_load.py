import runpy
import inspect

def pydict_load(filename, init_globals={}, verbose=False):
    if verbose:
        print('Loading dictionary from:', filename)

    dic = runpy.run_path(filename, init_globals)
    newdic = {}
    notthis={}
    for k, v in list(dic.items()):
        if k.startswith('__'):
            continue

        if v is init_globals.get(k, notthis):
            continue

        if inspect.ismodule(v):
            continue

        if inspect.isfunction(v):
            continue

        if inspect.ismethod(v):
            continue

        newdic[k] = v

    return newdic
