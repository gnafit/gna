Fitting with GNA
^^^^^^^^^^^^^^^^

We now will discuss how to fit a model using GNA UI. The minimal GNA setup, required to perform a fit includes a set of
items. Each item is usually defined by its own UI module.

The model
  An output, that depends on the minimization parameters. It will be fit to data.

    + The model usually comes with a set of parameters: free, constrained and fixed.

Experimental data
  An output, that will be used as data to fit the model to.

  This may be the same model, initialized with different versions of parameters and containing no fluctuations (i.e.
  Asimov dataset).

Dataset
  A dataset defines the connection between the model, data. It also may introduce nuisance terms based on the model
  parameters.

  At least one dataset should be defined. In the minimal configuration a dataset contains a single model and defines no
  nuisance parameters.

Analysis
  Analysis enables the user to combine several datasets (experiments). Analysis is also responsible for building a
  covariance matrix for the combined model.

  In the minimal configuration the analysis refers to a single dataset and defines no uncertainties, which should be
  propagated via covariance matrix.

Statistics
  A value of this function will be used to find best fit model. Statistics is usually a :math:`\chi^2`
  or Poisson function.

  Statistics uses an Analysis as input.

Minimizer
  A minimizer instance will be used to minimize the value of the Statistics. At the present time the Minuit minimizer is
  used from ROOT.

.. note::

    Nuisance terms and covariance matrix both define the uncertainty propagation procedure and therefore should contain
    no common parameters.

.. toctree::

    fitting_asimov
    fitting_using_snapshots
