import numpy as np
from minimize.lib.base import MinimizerBase, FitResult
import iminuit
from typing import Optional

class IMinuit(MinimizerBase):
    _label: str = 'iminuit'
    _minimizer: Optional[iminuit.Minuit] = None
    def __init__(self, statistic, minpars, **kwargs):
        MinimizerBase.__init__(self, statistic, minpars, **kwargs)

    def setuppars(self):
        self.update_minimizable()

        def fcn(x):
            x = np.ascontiguousarray(x, dtype='d')
            return self._minimizable(x)

        self._minimizer=iminuit.Minuit(fcn, self.parspecs.values(), name=self.parspecs.names())
        self._minimizer.throw_nan=True
        self._minimizer.errordef=1

        for i, (name, parspec) in enumerate(self.parspecs.items()):
            self.setuppar(i, name, parspec)

        self.parspecs.resetstatus()

    def setuppar(self, i, name, parspec):
        vmin, vmax = parspec.vmin, parspec.vmax
        if parspec.fixed:
            self._minimizer.fixed[i]=parspec.value

        self._minimizer.limits[i]=vmin, vmax
        self._minimizer.values[i]=parspec.value
        self._minimizer.errors[i]=parspec.step

    # def resetpars(self):
        # if self._reset:
            # return
        # spec = self.spec
        # for i, par in enumerate(self.parspecs):
            # self.setuppar(i, par, spec.get(par, {}))

    def _child_fit(self, profile_errors=[], covariance=False, scan=[]):
        assert self.parspecs
        import ROOT as R

        self.setuppars()
        with self.parspecs:
            fr = FitResult()
            try:
                with fr:
                    result=self._minimizer.migrad()
            except R.std.runtime_error as e:
                message = 'exception: '+e.what()
                success = False
                fun = None
                result=self._minimizer
            except RuntimeError:
                message = 'NaN'
                success = False
                fun = None
                result=self._minimizer
            else:
                success=result.valid
                message=str(result.fmin)
                fun=result.fval
            finally:
                argmin = np.array(result.values)
                errors = np.array(result.errors)
                fr.set(x=argmin, errors=errors, fun=fun,
                       success=success, message=message,
                       minimizer=self._label, nfev=result.nfcn
                       )

            self._result = fr.result
            self.patchresult()

            if self._result['success']:
                if covariance:
                    cov, status=self.get_covmatrix()
                    self._result['covariance']={'matrix': np.array(result.covariance), 'status': status}

                if profile_errors:
                    self.profile_errors(profile_errors, self.result)

                if scan:
                    self.get_scans(scan, self.result)

        return self.result

    def get_covmatrix(self, verbose=False):
        status = self._minimizer.fmin.has_covariance

        if verbose:
            if status:
                print('Covariance matrix:')
                print(covmatrix)
            else:
                print('Covariance matrix not estimated')

        if status:
            covmatrix = np.array(self._minimizer.covariance)
            return covmatrix, int(not status)

        return None, not status

    def get_scans(self, names, fitresult):
        import ROOT as R

        scans = fitresult['scan'] = {}
        if names:
            print('Caclulating profile for:', end=' ')
        for name in names:
            scan = scans[name] = {}
            try:
                xout, yout, valid = self._minimizer.mnprofile(name)
            except R.std.runtime_error:
                scan['x']=[]
                scan['y']=[]
                scan['success']=False
                scan['message']='runtime error'
            except RuntimeError:
                scan['x']=[]
                scan['y']=[]
                scan['success']=False
                scan['message']='NaN'
            else:
                scan['x']=xout.tolist()
                scan['y']=yout.tolist()
                scan['success']=valid.tolist()
                scan['message']=''

    def profile_errors(self, names, fitresult):
        import ROOT as R

        errs = fitresult['errors_profile'] = dict()
        statuses = fitresult['errors_profile_status'] = dict()
        if names:
            print('Caclulating profile error for:', names)

        for name in names:
            status = statuses[name] = dict()
            try:
                self._minimizer.minos(name)
            except R.std.runtime_error:
                errs[name] = [None, None]
                status['is_valid'] = False
                status['message'] = 'runtime error'
            except RuntimeError:
                errs[name] = [None, None]
                status['is_valid'] = False
                status['message'] = 'NaN'
            else:
                stat = self._minimizer.merrors[name]
                errs[name] = [stat.lower, stat.upper]

                for key in stat.__slots__:
                    status[key] = getattr(stat, key)
                status['message'] = ''

