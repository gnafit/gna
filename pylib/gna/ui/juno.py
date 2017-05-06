from gna.ui import basecmd

from gna.exp.reactor import ReactorExperimentModel, Reactor, Detector
import numpy as np
import ROOT

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        super(cmd, cls).initparser(parser, env)
        ReactorExperimentModel.initparser(parser, env)

    def init(self):
        common = {
            'power_rate': [1.0],
            'fission_fractions': {
                'U235':  [0.60],
                'Pu239': [0.27],
                'U238':  [0.07],
                'Pu241': [0.06],
            },
        }
        reactors_opts = [
            {'name': 'YJ1', 'location': 52.75, 'power': 2.9},
            {'name': 'YJ2', 'location': 52.84, 'power': 2.9},
            {'name': 'YJ3', 'location': 52.42, 'power': 2.9},
            {'name': 'YJ4', 'location': 52.51, 'power': 2.9},
            {'name': 'YJ5', 'location': 52.12, 'power': 2.9},
            {'name': 'YJ6', 'location': 52.21, 'power': 2.9},

            {'name': 'TS1', 'location': 52.76, 'power': 4.6},
            {'name': 'TS2', 'location': 52.63, 'power': 4.6},
            {'name': 'TS3', 'location': 52.32, 'power': 4.6},
            {'name': 'TS4', 'location': 52.20, 'power': 4.6},

            {'name': 'DYB', 'location': 215.0, 'power': 17.4},
            {'name': 'HZ', 'location': 265.0, 'power': 17.4},
        ]

        reactors = []
        for entry in reactors_opts:
            r = {}
            r.update(common)
            r.update(entry)
            reactors.append(Reactor(**r))

        detector = Detector(
            name='AD1',
            edges=np.linspace(1., 10., 200+1),
            location=0,
            protons=0.8*1.42e33, #TODO: is this detection efficiency of 80% ?
            livetime=[5*365*24*60*60.0],
        )

        ReactorExperimentModel(self.opts, reactors=reactors, detectors=[detector])
