Data inputs and fits
=======================

As for now we have considered only how to construct theoretical model
of an experiment. To carry out an analysis, at least two more pieces
are required: experimental data and uncertainties. Since no real
analysis was ever done inside the framework yet, the means to work
with that kind of inputs are rather rudimentary, not well-thought for
the real usage and not even tested. So, be ready and don't hesitate to
rewrite or extend anything.

The basic structure of the current design is the following:

- experimental data and uncertainties are just a numerical arrays in
  the form of output of a computational graph; even in the most simple
  case with constant values a computation node (e.g. ``Points`` class)
  should be constructed; in more complicated cases some intermediate
  transformations may be performed (like unfolding, extrapolation,
  etc) to get the data to analyze;
- experimental data and uncertainties are associated with a prediction
  and should have exactly the same shape; this is clearly targeted
  only to weighted least-squares analysis of binned counting data;
- to make analysis global, a covariance is defined between pair of
  experimental data by using the properly shaped covariance matrix;
  theoretical covariance is implemented by having common parameters
  between different experiments;
- the data, uncertainties and covariances associated with the
  corresponding observables form a *dataset* and are stored inside a
  ``gna.Dataset`` object; there may be many datasets for the same
  observations in arbitrary combinations;
- to analyze a set of specified observables they are concatenated
  into one linear sequence, as well as the data from the provided
  datasets; covariance matrices are constructed for each subset of
  non-covariated observables;
- a statistic is constructed from all that subsets that is later
  passed to minimization or scanning procedure.

Let's try to see how all of that works on some simple examples. First
example is to try to do simple fit of our Gaussian peak model to
hypothetical data, which we'll get from predictions with known
parameters and Poisson fluctuations. To get a significant event
number, let's set ``BackgroundRate`` to 10, ``Mu`` to 30 (just
arbitrary numbers for illustration) and bin numbers to 10, leaving
everything else default::

  $ python ./gna gaussianpeak --name peak --nbins 10 -- ns --value peak.BackgroundRate 10 
  --value peak.Mu 30 -- repl
  ...
  In [1]: self.env.get('peak/spectrum').data()
  Out[1]: 
  array([  5.08330775,  11.62487551,  11.62487551,   5.08330775,
           5.00000384,   5.        ,   5.        ,   5.        ,
           5.        ,   5.        ])

  In [2]: data =
  np.random.poisson(self.env.get('peak/spectrum').data())

  In [3]: data
  Out[3]: array([ 3, 11, 14,  2,  4,  3,  8,  3,  5,  1])

  In [4]: np.savez("/tmp/peakdata.npz", data=data)

Here we generate average theoretical spectrum for the specified
parameters and used ``np.random.poisson`` to make a Poisson random
sample out of it (of course you'll get different values, since it's
random). The data is saved in the .npz file ``/tmp/peakdata.npz``,
which we'll load during analysis.

Now we should forget about the parameters value, which we used to
generate a sample and use our estimation. In our toy example we'll
just say that we theoretically estimated ``BackgroundRate`` to be 14
with uncertainty of 30% and we want to find ``Mu`` by fitting. So, our
command will start with the model initialization::

  python ./gna ns --define peak.BackgroundRate central=14 relsigma=0.3 -- gaussianpeak
  --name peak --nbins 10
  
To load the data we can use repl or put the following commands to a file
(for example ``scripts/gaussianpeak_data``):

.. literalinclude:: examples/gaussianpeak_data.py
   :language: py
   :linenos:

Here we just read the data from the created npz file to variable ``x``
(line 4), create a new dataset with given description (line 5), then
assign ``x`` to be experimental values of the observable
``peak.spectrum``, and since the data is Poissonian, the same ``x``
goes as the experimental uncertainty (:math:`\sigma^2`), third argumnet
of assign). Finally, the dataset is stored in ``env`` under the name
``peak_fakedata`` for later usage. Ths script should be executed by
using the subcommand after ``gaussianpeak`` initialization::

  -- script scripts/gaussianpeak_data

Once we have created dataset and we are going to finalize inputs for
our analysis with the following command::

  -- analysis --name first_analysis --datasets peak_fakedata --observables peak/spectrum

The name of the analysis, which will be used later, is specified,
along with the datasets, whose data, uncertainties and covariance
information will be used and finally the observables that we are going to
analyze.

To turn the analysis into optimization problem, we need to use some
statistic. The only implemented for now the chi-squared (weighted
least squares) statistic. To use it, the following command should be
applied::

  -- chi2 first_analysis_chi2 first_analysis

where the name of the statistic (``first_analysis_chi2``) and the
corresponding analysis object to build statistic for
(``first_analysis``) are specified.

We are going to minimize our statistic, so minimizer should be
created::

 -- minimizer first_analysis_minimizer minuit first_analysis_chi2 peak.Mu
 peak.BackgroundRate

The arguments are the following: minimizer name, minimizer type (only
``minuit`` is implemented, check modules in the ``gna.minimizers``
package, then statistic for minimization and finally the list of
parameters to minimize over. Here we do two-dimensional minimization
over ``Mu`` and ``BackgroundRate``, all the other parameters are kept
fixed in their default values. Now, the last command is to issue the
minimization process::

  -- fit first_analysis_minimizer

Try to concatenate all commands into one string and run to get
the following result::

  Namespace(cpu=0.0068070000000002295, errors=array([ 11.56782544,   4.35315588]),
  fun=8.597269993657244, maxcv=0.01, nfev=35L, success=True,
  wall=0.006793022155761719, x=array([ 44.06893665,   5.1389115 ]))

This result is constructed directly from values returned by MINUIT,
the most important here are the following fields:

- ``success=True`` -- minimization procedure (MIGRAD) has converged,
- ``fun=8.597269993657244`` -- statistic value in found minimum,
- ``x=array([ 44.06893665,   5.1389115 ])`` -- minimizing parameter
  values, in the same order as passed to the ``minimizer`` command,
-  ``errors=array([ 11.56782544,   4.35315588])`` -- the corresponding
   uncertainties estimated by MINUIT.

We have reconstructed our parameters as ``Mu = 44.06893665 +-
11.56782544`` and ``BackgroundRate = 5.1389115 +- 4.35315588``. Not
very precise, but that's what we have.

It's always useful (when possible) to see the profile of the statistic
in form of the contour. To get one, we first need to scan the
statistic values over a set of points (i. e. grid). In our example we
can do two dimensional scan. First, we need to remove the parameters
which we are going to use in grid from the ``minimizer`` command::

  -- minimizer first_analysis_minimizer minuit

No parameters left, but that's fine. This minimizer does not actually
minimize anything just returning the statistic value in the current
point, but we need it to pass to the ``scan`` command::

  -- scan --grid peak.Mu 0 150 0.5 --grid peak.BackgroundRate 0 30 0.5 --minimizer
  first_analysis_minimizer --verbose --output /tmp/peak_scan.hdf5

First two arguments define the grids over two parameters. Few types of
grids can be used::

  - ``--grid parameter initial-value points-count step-size`` --
    constant step grid specified by initial value, points count and
    step size;
  - ``--lingrid parameter initial-value final-value points-count`` --
    constant step grid specified by initial value, final value and
    total points count including initial and final;
  - ``--loggrid parameter initial-value final-value points-count`` --
    constant step logarithmic (base 10) grid ``--lingrid``;
  - ``--listgrid parameter value1 value2 ...`` -- arbitrary spaced
    grid given just by any number of points.

All the grids parameter may be intermixed, the final grid will be
cartesian product over unions of grids for each parameter.

We also specify ``--minimizer``, which now just returns the statistic
value, the ``--verbose`` flag that's not required but will print the
values for each point during the scan and finally the output file.

When scan will done its work, you'll have the output file
``/tmp/peak_scan.hdf5``. You can explore it with ``hdf-java`` or
``h5dump``. It has so-called point-tree structure: for :math:`n`
parameters the first :math:`n` levels of hierarchy represent the
parameters value, the name of the :math:`i`-th level group is the
value of the :math:`i`-th parameter. The ordered list of the
parameters in stored in the ``params`` root attribute. On the
:math:`n+1`-th level we have the actual data -- the contents will
depend on concrete scan type, but generally there is ``datafit``
dataset where result for each specified minimizer is stored,
``fcparams`` where the minimizing parameters are stored (in the order
of the ``allparams`` root attribute). Just take a look and you'll see.

Finally let's plot the contour. To do that you need to get back the
minimizer with all parameters (it will be used to find the global
minimum) and issue the following command::

  -- contour --chi2 /tmp/peak_scan.hdf5 --plot chi2ci 1s 2s --minimizer
  first_analysis_minimizer --show

The first argument is a path to the chi2 values map, the second
specifies what we are going to plot -- `'chi2ci`` means chi-squared
confidence interval and this corresponds to the p-value given by the
:math:`\chi^2(n)` distribution. ``1s`` and ``2s`` means one and two
sigma, this will be mapped to the p-value for the two-tailed Gaussian.

The ``--show`` arguments asks ``matplotlib`` to show the plot on the
screen. If you like the plot, you can save it to a file with the
``--output path`` argument

In our two-dimensional fit we actually left the ``BackgroundRate``
completely free, no information about it's uncertainty was used. If we
want to take it into account we have two options, either add it ass
pull (punishment term) or add the corresponding covariation in
the covariation matrix.

The pull parameters in the framework are handled just like any real
experimental data -- the observable for them is the parameter value
itself, the data is the central value and the uncertainty is sigma,
specified in the parameter definition. Therefore, to add punishment
terms to the fit, we need to:

1. Create a new dataset for pull parameters; there is short command
   for that (check the code)::

     -- dataset --name pulls --pull peak.BackgroundRate

   Here just add parameter added to the dataset ``pulls`` (the name is
   arbitrary), any number can be specified.
2. Cdd the created dataset to the ``--datasets`` argument of the
   ``analysis`` command;
3. Add all the pull parameters to the ``--observations`` of the
   ``analysis`` command;
   
Try to construct the command yourself. Or here is the complete one
for reference::

  python ./gna ns --define peak.BackgroundRate central=14 relsigma=0.3 \
            -- gaussianpeak --name peak --nbins 10 \
            -- script scripts/gaussianpeak_data \
            -- dataset --name pulls --pull peak.BackgroundRate \
            -- analysis --name first_analysis --dataset peak_fakedata pulls \
            --observables peak/spectrum peak.BackgroundRate \
            -- chi2 first_analysis_chi2 first_analysis \
            -- minimizer first_analysis_minimizer minuit first_analysis_chi2 peak.Mu \
               peak.BackgroundRate \
            -- fit first_analysis_minimizer

It gives the following result::

  Namespace(cpu=0.0049440000000000595, errors=array([ 11.54509995,   1.10426956]),
  fun=12.74075442086563, maxcv=0.01, nfev=31L, success=True,
  wall=0.004951953887939453, x=array([ 42.59368222,   5.75145748]))

Just a bit better then the previous one, but that's expected since the
uncertainty is just 30\%.

Alternatively, you can follow the covariance matrix approach. Since
our theoretical model is linear with respect to ``BackgroundRate``,
the result should be exactly the same, as in the approach with pull
terms. Let's try. What's required is just to add the parameter to the
``--parameters`` argument of ``analysis`` it from minimization, and
dropping  the ``pulls`` dataset::

  python ./gna ns --define peak.BackgroundRate central=14 relsigma=0.3 \
            -- gaussianpeak --name peak --nbins 10 \
            -- script scripts/gaussianpeak_data \
            -- analysis --name first_analysis --dataset peak_fakedata \
            --observables peak/spectrum --parameters peak.BackgroundRate \
            -- chi2 first_analysis_chi2 first_analysis \
            -- minimizer first_analysis_minimizer minuit first_analysis_chi2 peak.Mu \
            -- fit first_analysis_minimizer

And the result is::

  Namespace(cpu=0.005394999999998262, errors=array([ 11.54509989]),
  fun=12.740754420865585, maxcv=0.01, nfev=13L, success=True,
  wall=0.005403995513916016, x=array([ 42.59368211]))

They match exactly, as expected, so everything works fine. Of course
if you have more than one nuisance parameters you can freely mix both
approaches. Just don't apply both on the same parameter, it doesn't
make sense.

Let's do one-dimensional scans and plots for both
approaches. Hopefully, you have an idea how to do it on the basis of
the previous commands. Here are the commands for scanning with pull::

  python ./gna ns --define peak.BackgroundRate central=14 relsigma=0.3 \
            -- gaussianpeak --name peak --nbins 10 \
            -- script scripts/gaussianpeak_data \
            -- dataset --name pulls --pull peak.BackgroundRate \
            -- analysis --name first_analysis --dataset peak_fakedata pulls \
            --observables peak/spectrum peak.BackgroundRate \
            -- chi2 first_analysis_chi2 first_analysis \
            -- minimizer first_analysis_minimizer minuit first_analysis_chi2 peak.BackgroundRate \
            -- scan --grid peak.Mu 0 150 0.5 --minimizer first_analysis_minimizer \
            --verbose --output /tmp/peak_scan_1d_pulls.hdf5

and covariance::

  python ./gna ns --define peak.BackgroundRate central=14 relsigma=0.3 \
            -- gaussianpeak --name peak --nbins 10 \
            -- script scripts/gaussianpeak_data \
            -- analysis --name first_analysis --dataset peak_fakedata \
            --observables peak/spectrum --parameters peak.BackgroundRate \
            -- chi2 first_analysis_chi2 first_analysis \
            -- minimizer first_analysis_minimizer minuit first_analysis_chi2  \
            -- scan --grid peak.Mu 0 150 0.5 --minimizer first_analysis_minimizer \
            --verbose --output /tmp/peak_scan_1d_covariance.hdf5
  
The plotting commands are also different because of different
minimizers for global minimization::

  python ./gna ns --define peak.BackgroundRate central=14 relsigma=0.3 \
            -- gaussianpeak --name peak --nbins 10 \
            -- script scripts/gaussianpeak_data \
            -- dataset --name pulls --pull peak.BackgroundRate \
            -- analysis --name first_analysis --dataset peak_fakedata pulls \
            --observables peak/spectrum peak.BackgroundRate \
            -- chi2 first_analysis_chi2 first_analysis \
            -- minimizer first_analysis_minimizer minuit first_analysis_chi2 peak.Mu \
               peak.BackgroundRate \
            -- contour --chi2 /tmp/peak_scan_1d_pulls.hdf5 --plot chi2ci 1s 2s \
            --minimizer first_analysis_minimizer --show

and::

  python ./gna ns --define peak.BackgroundRate central=14 relsigma=0.3 \
            -- gaussianpeak --name peak --nbins 10 \
            -- script scripts/gaussianpeak_data \
            -- analysis --name first_analysis --dataset peak_fakedata \
            --observables peak/spectrum --parameters peak.BackgroundRate \
            -- chi2 first_analysis_chi2 first_analysis \
            -- minimizer first_analysis_minimizer minuit first_analysis_chi2 peak.Mu \
            -- contour --chi2 /tmp/peak_scan_1d_covariance.hdf5 --plot chi2ci 1s 2s \
            --minimizer first_analysis_minimizer --show

I'm so sorry that the command line is verbose, please fix it whenever
possible.

And just a few words combined analysis. You can add as many datasets,
observables and parameters as you want, but this will clearly need
better interface than now. You can also add correlations between
experiments inside one dataset, the API is very simple::

  dataset.covariate(observable1, observable2, covariation_matrix)

The covariation matrix should be properly shaped (:math:`m \times n`,
where :math:`m` and :math:`n` are dimensions of ``observable1`` or
``observable2`` correspondingly). ``covariation_matrix`` may be just a
numpy array with numbers or computational graph output. All applicable
covariations from the specified datasets will be used by
``analysis``. This is not well tested though, so expect a lot of
surprises and debugging. The only usage is in the
``gna.ui.oldreactors``, please refer to it.
