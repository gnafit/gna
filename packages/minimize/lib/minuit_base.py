import ROOT
import numpy as np
from minimize.lib.base import MinimizerBase, FitResult

class MinuitBase(MinimizerBase):
    _label = 'TMinuit?'
    def __init__(self, statistic, minpars, **kwargs):
        MinimizerBase.__init__(self, statistic, minpars, **kwargs)

        print("\033[35mWarning! Disowning the C++ minimizer class to avoid segfault. Issue #138 of !67.\033[0m")
        ROOT.SetOwnership(self, False)

    @property
    def tolerance(self):
        return self.Tolerance()

    @tolerance.setter
    def tolerance(self, tolerance):
        self.SetTolerance(tolerance)

    @property
    def label(self):
        return self._label

    def setuppars(self):
#         Comment out for good working of minimizer-v1
#         Issue #148(https://git.jinr.ru/gna/gna/-/issues/148)
#         if not self.parspecs.modified:
#             return

        self.update_minimizable()

        self.SetFunction(self._minimizable)

        for i, (name, parspec) in enumerate(self.parspecs.items()):
            self.setuppar(i, name, parspec)

        self.parspecs.resetstatus()

    def setuppar(self, i, name, parspec):
        vmin, vmax = parspec.vmin, parspec.vmax
        if parspec.fixed:
            self.SetFixedVariable(i, name, parspec.value)
        elif (vmin, vmax) == (None, None):
            self.SetVariable(i, name, parspec.value, parspec.step)
        elif vmax is None:
            self.SetLowerLimitedVariable(i, name, parspec.value, parspec.step, vmin)
        elif vmin is None:
            self.SetUpperLimitedVariable(i, name, parspec.value, parspec.step, vmax)
        else:
            self.SetLimitedVariable(i, name, parspec.value, parspec.step, vmin, vmax)

    # def resetpars(self):
        # if self._reset:
            # return
        # spec = self.spec
        # for i, par in enumerate(self.parspecs):
            # self.setuppar(i, par, spec.get(par, {}))

    def _child_fit(self, profile_errors=[], covariance=False):
        assert self.parspecs

        self.setuppars()
        with self.parspecs:
            with FitResult() as fr:
                success=self.Minimize()

            argmin = np.frombuffer(self.X(), dtype=float, count=self.NDim())
            errors = np.frombuffer(self.Errors(), dtype=float, count=self.NDim())
            status = self.Status()
            fr.set(x=argmin, errors=errors, fun=self.MinValue(),
                   success=success, message='{} ({})'.format(StatusCodes.get(status, StatusCodes['unknown']), status),
                   minimizer=self.label, nfev=int(self.NCalls())
                    )
            self._result = fr.result
            self.patchresult()

            if covariance:
                cov, status=self.get_covmatrix()
                self._result['covariance']={'matrix': cov, 'status': status}

            if profile_errors:
                self.profile_errors(profile_errors, self.result)

        return self.result

    def profile_errors(self, names, fitresult):
        errs = fitresult['errors_profile'] = dict()
        statuses = fitresult['errors_profile_status'] = dict()
        if names:
            print('Caclulating profile error for:', end=' ')
        for name in names:
            if isinstance(name, int):
                idx = name
                name = self.VariableName(idx)
            else:
                idx = self.result['names'].index(name)
            print(name, end=', ')
            status, (left, right) = self.get_profile_error(idx=idx)
            errs[name] = [left, right]
            statuses[name] = status

    def get_profile_error(self, name=None, idx=None, verbose=False):
        if idx==None:
            idx = self.VariableIndex( name )

        if not name:
            name = self.VariableName( idx )

        if verbose:
            print( '    variable %i %s'%( idx, name ), end='' )

        low, up = np.zeros( 1, dtype='d' ), np.zeros( 1, dtype='d' )
        try:
            ret = self.GetMinosError( idx, low, up )
        except:
            ret = False

        minosstatus = (self.Status()%100)//10

        if not ret:
            print( 'Minuit error!' )
            return minosstatus, [ 0.0, 0.0 ]

        print( ':', low[0], up[0] )
        return minosstatus, [ low[0].tolist(), up[0].tolist() ]

    def get_covmatrix(self, verbose=False):
        n = self.NDim()

        if verbose:
            print( '    calculating covariance matrix...' )

        covmatrix = np.zeros((n, n), dtype='d')
        self.GetCovMatrix(covmatrix.ravel())
        if verbose:
            print( covmatrix )
        status = self.CovMatrixStatus()

        return covmatrix, status

StatusCodes = {
        0: 'Minimization converged',
        1: 'Covar was made pos def',
        2: 'Hesse is not valid',
        3: 'Edm is above max',
        4: 'Reached call limit',
        5: 'Covar is not pos def',
        6: "unknown failure",
        'unknown': 'Unknown status code'
}
