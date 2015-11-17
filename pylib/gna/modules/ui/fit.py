from gna.ui import basecmd

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser):
        parser.add_argument('minimizer')

    def init(self):
        minimizer = self.env.minimizers[self.opts.minimizer]
        print minimizer.fit()

