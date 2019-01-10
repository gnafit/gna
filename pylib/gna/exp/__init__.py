class baseexp(object):
    @classmethod
    def initparser(cls, parser, env):
        pass

    def __init__(self, env, opts):
        self.env  = env
        self.opts = opts

