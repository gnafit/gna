from load import ROOT as R
import numpy as np
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import StripNestedDict
from tools.schema import Schema, Optional, And, isfilewithext, haslength

class detector_energy_leakage_root_v02(TransformationBundle):
    """Energy distortion due to the energy leakage (v02), defined via a histogram from a ROOT file

    Updates since detector_energy_leakage_root_v01:
        - schema evaluation
        - optional usage of triangular matrix

    Configuration fields:
        - filename - name of the ROOT file
        - matrixname - name of the matrix in a file
        - matrix_is_upper [default=False] - use only upper part of the matrix if True

    Optional configuration fields:
        - par - dictionary with uncertain parameter options:
            - name - nuisance parameter name (string)
            - scale - 'uncertain' with the definition of uncertainty
            - ndiag=1 - number of diagonals to apply scale factor (with uncertainty) to

    Provided variables:
        - parname - uncertain parameter name

    Provided outputs:
        - 'eleak_matrix_raw' - raw energy leakage matrix (in case of nuisance parameter)
        - 'eleak_matrix' - energy leakage matrix after scaling the diagonal
        - 'eleak' - the result of the distortion

    Provided inputs:
        - 'eleak' - an input to apply the distortion
    """
    eleak_matrix=None
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 1, 'major')

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

    _validator = Schema({
            'bundle': object,
            'filename': isfilewithext('root'),
            'matrixname': str,
            Optional('matrix_is_upper', default=False): bool,
            Optional('par'): {
                    'name': str,
                    'scale': object,
                    'ndiag': int,
                },
            Optional('pad', default=None): And((int,), haslength(exactly=2)) # pad the matrix with zeros: see `numpy.pad`
        })


    @staticmethod
    def _provides(cfg):
        parname = cfg.get('par', {}).get('name')
        if parname:
            return (parname,), ('eleak_matrix_raw', 'eleak_matrix', 'eleak')

        return (), ('eleak_matrix', 'eleak')

    def build_mat(self):
        """Assembles a chain for eleak detector effect using input matrix"""
        haspar = 'par' in self.vcfg
        if haspar:
            ndiag = self.vcfg['par']['ndiag']
            parname = self.vcfg['par']['name']
        matrix_is_upper = self.vcfg['matrix_is_upper']
        smtype = R.GNA.SquareMatrixType.UpperTriangular if matrix_is_upper else R.GNA.SquareMatrixType.Any

        if matrix_is_upper:
            self.eleak_matrix = np.triu(self.eleak_matrix)

        pad = self.vcfg['pad']
        if pad is not None:
            self.eleak_matrix = np.pad(self.eleak_matrix, pad_width=pad)

        norm = self.eleak_matrix.sum( axis=0 )
        norm[norm==0.0]=1.0
        self.eleak_matrix/=norm

        points = C.Points(self.eleak_matrix, ns=self.namespace, labels='Eleak matrix\n raw')
        self.context.objects['matrix'] = points
        if haspar:
            self.set_output('eleak_matrix_raw', None, points.single())
        else:
            self.set_output('eleak_matrix', None, points.single())

        for itdet in self.nidx_major:
            if haspar:
                parname = itdet.current_format(name=parname)
                # arg0: ndiag, arg1: Target=OffDiagonal, arg2: Mode=Upper
                renormdiag = R.RenormalizeDiag(ndiag, 1, int(matrix_is_upper), parname, ns=self.namespace, labels=itdet.current_format('Eleak matrix\n {autoindex}'))
                points.points >> renormdiag.renorm.inmat
                self.set_output('eleak_matrix', itdet, renormdiag.single())
                self.context.objects[itdet.current_values(name='renormdiag')] = renormdiag
                matinput = renormdiag.renorm
            else:
                matinput = points.points

            for itother in self.nidx_minor:
                it = itdet+itother
                esmear = R.HistSmear(smtype, labels=it.current_format('{{Eleak effect|{autoindex}}}'))
                matinput >> esmear.smear.inputs.SmearMatrix
                self.set_input('eleak', it, esmear.smear.Ntrue, argument_number=0)
                self.set_output('eleak', it, esmear.single())

                self.context.objects[it.current_values(name='esmear')] = esmear

    def build(self):
        from tools.data_load import read_object_auto
        res = read_object_auto(self.vcfg['filename'], name=self.vcfg['matrixname'])
        if isinstance(res, tuple):
            self.eleak_matrix = res[-1]
        else:
            self.eleak_matrix = res

        return self.build_mat()

    def define_variables(self):
        paropts = self.vcfg.get('paropts')
        if not paropts:
            return
        parname = paropts['name']
        scale = paropts['scale']
        if scale.mode!='relative':
            raise Exception('Eleak uncertainty should be relative by definition')
        if scale.central!=1.0:
            raise exception('Eleak scale should be 1 by definition')

        for it in self.nidx_major:
            self.reqparameter(parname, it, cfg=scale, label='Eleak offdiagonal contribution scale at {autoindex}')

