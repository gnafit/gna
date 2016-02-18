import numpy as np
from gna.ui import basecmd
from gna.exp.reactor import ReactorExperimentModel, Reactor, Detector, Isotope
import gna.parameters.oscillation
from gna.dataset import Dataset
from argparse import Namespace

class Data2Theory(object):
    def __init__(self,name,fission,L,value,error,corr_exp,corr_theory):
        if name == None:
            assert name, 'A name should be provided'
        self.name = name
        self.fission = fission
        self.L = L
        self.value = value
        self.error = error
        self.corr_exp = corr_exp
        self.corr_theory = corr_theory
        #print self.name, 'created'    
    #end def __init__
#

data2theory = []

data2theory.append( Data2Theory('ILL',  {'U235': 0.620, 'Pu239': 0.274, 'U238': 0.074, 'Pu241': 0.042}, 
                                8.76, 0.797, 9.0, {'ILL':9.0, 'Gosgen-I':4.8, 'Gosgen-II':4.8, 'Gosgen-III':4.8}, 0) )

data2theory.append( Data2Theory('Gosgen-I',  {'U235': 0.620, 'Pu239': 0.274, 'U238': 0.074, 'Pu241': 0.042}, 
                                37.9, 0.960, 5.36, {'Gosgen-I':5.36, 'Gosgen-II':4.8, 'Gosgen-III':4.8, 'ILL':4.8}, 0) )

data2theory.append( Data2Theory('Gosgen-II', {'U235': 0.584, 'Pu239': 0.298, 'U238': 0.068, 'Pu241': 0.050},
                                45.9, 0.986, 5.33, {'Gosgen-I':4.8, 'Gosgen-II':5.33, 'Gosgen-III':4.8, 'ILL':4.8}, 0) )

data2theory.append( Data2Theory('Gosgen-III', {'U235': 0.543, 'Pu239': 0.329, 'U238': 0.070, 'Pu241': 0.058},
                                64.7, 0.919, 6.80, {'Gosgen-I':4.8, 'Gosgen-II':4.8, 'Gosgen-III':6.80, 'ILL':4.8}, 0) )

data2theory.append( Data2Theory('Krasnoyarsk-I', {'U235': 1.0, 'Pu239': 0.0, 'U238': 0.0, 'Pu241': 0.0},
                                32.8, 0.930, 6.00, {'Krasnoyarsk-I':6.0, 'Krasnoyarsk-II':4.84, 'Krasnoyarsk-III':4.84, 
                                                    'Krasnoyarsk-IV':4.84}, 0) )

data2theory.append( Data2Theory('Krasnoyarsk-II', {'U235': 1.0, 'Pu239': 0.0, 'U238': 0.0, 'Pu241': 0.0},
                                92.3, 0.947, 20.40, {'Krasnoyarsk-I':4.76, 'Krasnoyarsk-II':20.4, 'Krasnoyarsk-III':4.76, 
                                                    'Krasnoyarsk-IV':4.76}, 0) )

data2theory.append( Data2Theory('Krasnoyarsk-III', {'U235': 1.0, 'Pu239': 0.0, 'U238': 0.0, 'Pu241': 0.0},
                                57.0, 0.941, 4.15, {'Krasnoyarsk-I':3.4, 'Krasnoyarsk-II':3.4, 'Krasnoyarsk-III':4.4, 
                                                    'Krasnoyarsk-IV':3.4}, 0) )

data2theory.append( Data2Theory('Krasnoyarsk-IV', {'U235': 1.0, 'Pu239': 0.0, 'U238': 0.0, 'Pu241': 0.0},
                                231.4, 1.093, 17.75, {'Krasnoyarsk-I':3.4, 'Krasnoyarsk-II':3.4, 'Krasnoyarsk-III':3.4, 
                                                    'Krasnoyarsk-IV':17.75}, 0) )

data2theory.append( Data2Theory('Rovno88-1I', {'U235': 0.607, 'Pu239': 0.277, 'U238': 0.074, 'Pu241': 0.042},
                                18.0, 0.911, 6.39, {'Rovno88-1I':6.39, 'Rovno88-2I':2.2,'Rovno88-1S':2.2,
                                                    'Rovno88-2S':2.2,'Rovno88-3S':2.2}, 0) )
                                
data2theory.append( Data2Theory('Rovno88-2I', {'U235': 0.603, 'Pu239': 0.276, 'U238': 0.076, 'Pu241': 0.045},
                                17.96, 0.942, 6.39, {'Rovno88-1I':2.2, 'Rovno88-2I':6.39,'Rovno88-1S':2.2,
                                                    'Rovno88-2S':2.2,'Rovno88-3S':2.2}, 0) )
data2theory.append( Data2Theory('Rovno88-1S', {'U235': 0.606, 'Pu239': 0.277, 'U238': 0.074, 'Pu241': 0.043},
                                18.15, 0.966, 7.34, {'Rovno88-1I':2.2, 'Rovno88-2I':2.2,'Rovno88-1S':7.34,
                                                    'Rovno88-2S':2.2,'Rovno88-3S':2.2}, 0) )
data2theory.append( Data2Theory('Rovno88-2S', {'U235': 0.557, 'Pu239': 0.313, 'U238': 0.076, 'Pu241': 0.054},
                                25.17, 0.953, 7.34, {'Rovno88-1I':2.2, 'Rovno88-2I':2.2,'Rovno88-1S':2.2,
                                                    'Rovno88-2S':7.34,'Rovno88-3S':2.2}, 0) )

data2theory.append( Data2Theory('Rovno88-3S', {'U235': 0.606, 'Pu239': 0.274, 'U238': 0.074, 'Pu241': 0.046},
                                18.18, 0.932, 6.77, {'Rovno88-1I':2.2, 'Rovno88-2I':2.2,'Rovno88-1S':2.2,
                                                    'Rovno88-2S':2.2,'Rovno88-3S':6.77}, 0) )

data2theory.append( Data2Theory('Rovno91', {'U235': 0.614, 'Pu239': 0.274, 'U238': 0.074, 'Pu241': 0.038},
                                18.19, 0.934, 2.8, {'Rovno91':2.8, 'Bugey-4':1.8, 'Rovno92':2.8}, 0) )

data2theory.append( Data2Theory('Rovno92', {'U235': 0.614, 'Pu239': 0.274, 'U238': 0.074, 'Pu241': 0.038},
                                18.19, 0.912, 3.8, {'Rovno92':3.8, 'Rovno91':2.8, 'Bugey-4':2.8}, 0) )

data2theory.append( Data2Theory('Bugey-4', {'U235': 0.538, 'Pu239': 0.328, 'U238': 0.078, 'Pu241': 0.056},
                                14.882, 0.936, 1.4, {'Bugey-4':1.4, 'Rovno91': 0.84, 'Rovno92': 0.84}, 0) )

data2theory.append( Data2Theory('Bugey-3-I', {'U235': 0.538, 'Pu239': 0.328, 'U238': 0.078, 'Pu241': 0.056},
                                15.0, 0.940, 4.45, {'Bugey-3-I':4.45, 'Bugey-3-II':4.0, 'Bugey-3-III':4.0}, 0) )

data2theory.append( Data2Theory('Bugey-3-II', {'U235': 0.538, 'Pu239': 0.328, 'U238': 0.078, 'Pu241': 0.056},
                                40.0, 0.946, 4.54, {'Bugey-3-I':4.0, 'Bugey-3-II':4.54, 'Bugey-3-III':4.0}, 0) )

data2theory.append( Data2Theory('Bugey-3-III', {'U235': 0.538, 'Pu239': 0.328, 'U238': 0.078, 'Pu241': 0.056},
                                95.0, 0.870, 15.1, {'Bugey-3-I':4.0, 'Bugey-3-II':4.0, 'Bugey-3-III':15.1}, 0) )

data2theory.append( Data2Theory('SRP-I', {'U235': 1, 'Pu239': 0.0, 'U238': 0.0, 'Pu241': 0.0},
                                18.18, 0.946, 2.40, {'SRP-I':1.5, 'SRP-II':1.5}, 0) )

data2theory.append( Data2Theory('SRP-II', {'U235': 1, 'Pu239': 0.0, 'U238': 0.0, 'Pu241': 0.0},
                                23.82, 1.011, 2.53, {'SRP-I':1.5, 'SRP-II':1.5}, 0) )

data2theory.append( Data2Theory('Chooz', {'U235': 0.496, 'Pu239': 0.351, 'U238': 0.087, 'Pu241': 0.066},
                                1050.0, 0.955, 3.39, {'Chooz':3.39}, 0) )

data2theory.append( Data2Theory('Palo Verde', {'U235': 0.60, 'Pu239': 0.27, 'U238': 0.07, 'Pu241': 0.06},
                                820.0, 0.969, 5.43, {'Palo Verde':5.43}, 0) )

data2theory.append( Data2Theory('Double Chooz (Gd)', {'U235': 0.496, 'Pu239': 0.351, 'U238': 0.087, 'Pu241': 0.066},
                                1050.0, 0.881, 2.3, {'Double Chooz (Gd)':2.3}, 0) )

data2theory.append( Data2Theory('Double Chooz (n)', {'U235': 0.496, 'Pu239': 0.351, 'U238': 0.087, 'Pu241': 0.066},
                                1050.0, 0.906, 2.9, {'Double Chooz (n)':2.9}, 0) )

data2theory.append( Data2Theory('KamLAND', {'U235': 0.571, 'Pu239': 0.295, 'U238': 0.078, 'Pu241': 0.056},
                                175000.0, 0.589, 15.87, {'KamLAND':15.87}, 0) )

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        ReactorExperimentModel.initparser(parser, env)
        parser.add_argument('--name', default='oldreactor')

    def run(self):
        dataset = Dataset(self.opts.name, "old reactors world data")
        self.env.parts.dataset[self.opts.name] = dataset

        ns = self.env.ns("common")
        ReactorExperimentModel.reqparameters(ns)

        observables = {}
        for i, expdata in enumerate(data2theory):
            expname = "exp{}".format(i)
            expns = self.env.ns(expname)
            expns.inherit(ns)

            reactors = [Reactor(expns,
                        name='reactor',
                        location=expdata.L/1e3,
                        fission_fractions={isoname: [frac] for isoname, frac in expdata.fission.iteritems()})]
            detectors = [Detector(expns, name='AD1', location=0)]

            expns.defparameter("Norm", central=1, sigma=0)
            with expns:
                exp = ReactorExperimentModel(self.opts, expname, ns=expns, reactors=reactors, detectors=detectors)
            expns["Norm"].set(1/expns.observables['AD1_comp0'].data()[0])
            observables[expdata] = expns.observables['AD1']

        covariated = set()
        for expdata1, obs1 in observables.iteritems():
            dataset.assign(observables[expdata1], [expdata1.value], [expdata1.value*expdata1.error/1e2])
            for corrname, corrcoeff in expdata1.corr_exp.iteritems():
                expdata2 = next(x for x in data2theory if x.name == corrname)
                obs2 = observables[expdata2]
                if obs1 == obs2 or covariated & {(obs1, obs2), (obs2, obs1)}:
                    continue
                covariated.add((obs1, obs2))
                dataset.covariate(obs1, obs2, [expdata1.value*expdata2.value*corrcoeff/1e2])
