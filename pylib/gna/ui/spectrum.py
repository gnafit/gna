from gna.ui import basecmd, append_typed, qualified
from matplotlib import pyplot as plt
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        def observable(path):
            nspath, name = path.split('/')
            try:
                return env.ns(nspath).observables[name]
            except KeyError:
                raise PartNotFoundError("observable", path)

        parser.add_argument('-p', '--plot', default=[],
                            metavar=('DATA',),
                            action=append_typed(qualified(env.parts.prediction, observable)))

    def run(self):
        ax = plt.gca()
        for plotobj in self.opts.plot:
            data = plotobj.data()
            edges = plotobj.datatype().hist().edges()
            if (edges.size()-1,) != data.shape:
                msg = "edges shape mismatch for 1d histogram: {0!r} vs {1!r}"
                raise Exception(msg.format((edges.size()-1,), data.shape))
            edges = np.array(edges)
            ax.plot((edges[:-1] + edges[1:])/2, data)
        plt.show()
