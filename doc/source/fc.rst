A few notes on Feldman-Cousins
=================================

Let's try to do the same contours with the previous example within
Feldman-Cousins approach. The only thing that's needed in addition to
the statistic map produced in the previous example is the distribution
of the statistic with respect to fluctuations of the theoretical
expectation on each point of the grid. To construct such a map, we
need to employ ToyMC -- just simple generator of fake experimental
data from average prediction. Currently two such ToyMCs are
implemented namely ``CovariatedToyMC`` which samples from Gaussian
according to the covariance matrix and ``PoissonToyMC`` which
samples Poisson distribution just by using prediction as the
average. More is to be implemented for real like analysis, for
example to sample nuissance parameters from some distribution.

We can try to employ both of them. To do that we just need to supply
``--toymc`` to the ``analysis`` command, which will just replace the
real data with the corresponding ToyMC generator. As the argument to
``--toymc`` you may pass ``covariance`` or ``poisson``
correspondingly.

Then it's also required to adjust the ``scan`` command. We need to
add here ``--toymc`` too, but the argument means the name of ``toymc``
object (the same as of the ``analysis``). Also we need the
``--toymc-type grid`` -- this instructs the ``scan`` to generate
different samples on each point of the grid (the other option is
``--toymc-type static`` is useful for example for coverage check or
CLs, but it needs some debugging). Another difference is that we need
to provide two minimizers -- one for used for the minimum with the
parameter value fixed on grid, the other is the global
minimization. Finally, samples count to generate is specified with
``--samples`` (we choose 10000).

Here is the full command for one-dimensional (``BackgroundRate``
uncertainty is taken into account in the covariance matrix) scan for
CovarianceToyMC:

.. code-block:: bash

  python ./gna ns --define peak.BackgroundRate central=14 relsigma=0.3 \
            -- gaussianpeak --name peak --nbins 10 \
            -- script scripts/gaussianpeak_data \
            -- dataset --name pulls --pull peak.BackgroundRate \
            -- analysis --name first_analysis --dataset peak_fakedata \
            --observables peak/spectrum --parameters peak.BackgroundRate --toymc covariance \
            -- chi2 first_analysis_chi2 first_analysis \
            -- minimizer ongrid_minimizer minuit first_analysis_chi2 \
            -- minimizer global_minimizer minuit first_analysis_chi2 peak.Mu \
            -- scan --output /tmp/peak_fc_covariance.hdf5 --grid peak.Mu 0 150 0.5 \
            --toymc first_analysis --toymc-type grid --samples 10000 \
            --minimizer ongrid_minimizer --minimizer global_minimizer

Give it some time to finish its work. Then check it with ``hdf-java``
or ``h5dump`` and you'll see the similar structure as for normal
chi-square map, but with a lot (10000) of entries in each point
corresponding to different random samples.

The last step before the plotting is to generate so-called ``fcmap``
which is basically just sorted list of :math:`\Delta \chi^2` values
for quantile computation. The command is very simple:

.. code-block:: bash

  python ./gna fcmap --fcscan /tmp/peak_fc_covariance.hdf5 --output /tmp/peak_fcmap_covariance.hdf5

Assuming you have ``/tmp/peak_scan_1d_covariance.hdf5`` from the
previous part, let's run the plotting command:

.. code-block:: bash

  python ./gna ns --define peak.BackgroundRate central=14 relsigma=0.3 \
            -- gaussianpeak --name peak --nbins 10 \
            -- script scripts/gaussianpeak_data \
            -- analysis --name first_analysis --dataset peak_fakedata \
            --observables peak/spectrum --parameters peak.BackgroundRate \
            -- chi2 first_analysis_chi2 first_analysis \
            -- minimizer first_analysis_minimizer minuit first_analysis_chi2 peak.Mu \
            -- contour --chi2 /tmp/peak_scan_1d_covariance.hdf5 --plot chi2ci 1s 2s \
            --minimizer first_analysis_minimizer --fc /tmp/peak_fcmap_covariance.hdf5 \
            --plot fc 1s 2s --show

In comparison to the previous plotting, we have just added ``--fc``
with the path of the fcmap and options to ``--plot fc``. You should
see nice picture -- two curves almost match (up to
fluctuations). That's natural, because our case is very good -- no
constraints on parameter, everything is Gaussian-distributed. Try the
same excercise with ``PoissonToyMC`` (just change ``--toymc
covariance`` to  ``--toymc poisson`` in the ``analysis`` and the
filenames) and you'll see completely different picture -- it seems
like just a lower limit can be obtained. No gurantee about the
correctness of the conclusion though.

Another interesting thing is to see how limits on the fitting
parameter influences the contour. To do that, you need to use so
called minimizer spec -- a short description about parameters in YAML
format. It's passed to the ``minimizer`` command with ``-s``
option. To limit ``peak.Mu`` in ``[0; 100]`` bounds, modify all
minimizers with ``peak.Mu`` in the following way:
  
.. code-block:: bash

  -- minimizer -s '{peak.Mu: {limits: [0, 100]}}' global_minimizer minuit \
  first_analysis_chi2 peak.Mu

Don't forget to check the details (and look for other possible
options) in ``gna.minimizers.spec``.

To make two-dimensional Feldman-Cousins contours you can use all the
same machinery, just specify the correct parameters in minimizers and
corresponding grids. But to save some computation time, you also can
try to produce fcmap only in region where you expect your contour to
be, for example by selecting the points of the standard chi-squared
contour plus some margin. It can be done by saving the interesting
points list to a file and giving it to the ``scan`` as input with the
``--points`` argument. The points list may be produced by the
``contour``. For example:

.. code-block:: bash

  contour --chi2 /tmp/peak_scan.hdf5 --plot chi2ci 1s 2s --minimizer \
  first_analysis_minimizer  --show --points chi2ci 1s --savepoints /tmp/peak_points.hdf5

The given command will save to the file ``/tmp/peak_points.hdf5`` the
points around 1 sigma contour (check it in ``hdf-java``). The width of
band around the contour may be controlled by two additional numbers
to ``--points`` -- inside and outside width. They have no definite
meaning, just larger value means wider band in the corresponding
direction. The default values are ``0.05 0.05``. When the file is
generated, just pass its path to the ``scan`` instead of all the
``--grid``-s.

Finally, you will definitely want to run the Feldman-Cousins scanning
on a cluster utilizing a lot of CPU cores. You can efficiently split
the tasks by the points utilizing the ``--pointsrange`` argument to
the ``scan``. It takes up to three integer, which are interpreted as
python indexing or slice: ``[a]``, ``[a:b]`` or ``[a:b:c]`` on the
linearized list of all points in file, given by
``--points``. Alternatively, you can specify interesting points
directly by given several points path in form ``--pointspath path1
path2 ...``, where path is just numerical values of parameters (in the
same order as in the points file). The points file is still
required. Finally, you can split by samples number. After you'll get a
lot of splitted scan results, you should merge them into one file
before processing with ``fcmap`` by using ``fcmerge``:

.. code-block:: bash
   
  python ./gna fcmerge --fcscan fcfile1 fcfile2 ... --output mergedfile

where ``fcfile``-s are partial scan outputs and ``mergedfile`` is the
final output to be passed to ``fcmap``.
