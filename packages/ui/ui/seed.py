"""
Set seed of randomization.

Can be used to reproduce results of MC simulations.
"""

from gna.ui import basecmd
import ROOT
import numpy as np

class cmd(basecmd):
    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-s', '--seed', type=int, required=True, help='set seed of randomization (BOOST)')
        parser.add_argument('-n', '--seed-np', type=int, help='set seed of randomization (numpy)')
        parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')

    def init(self):
        ROOT.GNA.Random.seed(self.opts.seed)

        if self.opts.seed_np is not None:
            np.random.seed(self.opts.seed_np)

        if self.opts.verbose:
            print(f'Set BOOST random seed: {self.opts.seed}')
            if self.opts.seed_np is not None:
                print(f'Set numpy random seed: {self.opts.seed_np}')

    __tldr__ = """\
                The main argument is the number to be setted as seed, which will be used for the BOOST random generator.

                Set 10 as seed:
                ```sh
                ./gna \\
                    -- seed -s 10
                ```

                A seed for the numpy random generator may be assinged optionally with:
                ```sh
                ./gna \\
                    -- seed -s 10 -n 11
                ```

                In the verbose mode it also prints the command to the stdout.
                """

