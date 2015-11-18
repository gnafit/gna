class basecmd(object):
    @classmethod
    def initparser(cls, parser):
        pass

    def __init__(self, args):
        self.opts = args

    def init(self):
        pass

    def run(self):
        pass
