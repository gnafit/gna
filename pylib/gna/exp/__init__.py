class baseexp(object):
    @classmethod
    def initparser(cls, parser, namespace):
        pass

    def __init__(self, namespace, opts):
        self.namespace  = namespace
        self.opts = opts

    def _exception(self, message):
        return RuntimeError(f'{self.__class__.__name__} error: {message}')

