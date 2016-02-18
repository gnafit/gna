from gna.exp.reactor import ReactorExperimentModel, Reactor, Detector
import numpy as np
import ROOT

class exp(ReactorExperimentModel):
    name = 'juno'

    def makereactors(self):
        common = {
            'power_rate': [1.0],
            'isotopes': ReactorExperimentModel.makeisotopes(self.ns),
            'fission_fractions': {
                'Pu239': [0.60],
                'Pu241': [0.07],
                'U235': [0.27],
                'U238': [0.06],
            }
        }
        data = [
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
        return [Reactor(self.ns, **dict(common.items(), **x)) for x in data]

    def makedetectors(self):
        detector = Detector(
            self.ns,
            name='AD1',
            location=0,
            protons=1.42e33,
            livetime=[6*365*24*60*60.0]
        )
        return [detector]
