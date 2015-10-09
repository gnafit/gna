from collections import defaultdict
from gna.parameters import parameters

class _environment(object):
    def __init__(self):
        self.pars = parameters()
        self.objs = []
    def register(self, obj, **kwargs):
        if len(obj.evaluables):
            self.pars.addexpressions(obj, bindings=kwargs.get("bindings", {}))
        if not kwargs.pop('bind', True):
            return obj
        self.pars.resolve(obj, **kwargs)
        self.objs.append(obj)
        return obj

class _environments(defaultdict):
    current = None

    def __init__(self):
        super(_environments, self).__init__(_environment)

    def __getattr__(self, name):
        return self[name]

    def __getitem__(self, name):
        env = super(_environments, self).__getitem__(name)
        if _environments.current is None:
            _environments.current = env
        return env

env = _environments()
