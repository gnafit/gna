Fitting to Asimov dataset
"""""""""""""""""""""""""

Let us now construct an example setup for fitting. We will use `gaussianpeak` module to provide both the data and the
model to be fit.

At first, we initialize two `gaussianpeak` modules with names `peak_MC` and `peak_f`. The former will be used as data
while the latter will be used as a model to be fit.

.. literalinclude:: ../../../macro/tutorial/fit/01_fit_script.sh
    :linenos:
    :lines: 3-5
    :caption: :download:`01_fit_script.sh <../../../macro/tutorial/fit/01_fit_script.sh>`

Now, using `ns` module we change the parameters for the models:

.. literalinclude:: ../../../macro/tutorial/fit/01_fit_script.sh
    :linenos:
    :lines: 6-15

Here we have used slightly different approach to the parameters. Instead of defining the parameters prior the models, we
change them after the module creates the default versions. Syntax of the :code:`--set` option is pretty similar to the
one of the :code:`--define`.

This command line produces the following output.

.. code-block:: text

   Add observable: peak_MC/spectrum
   Add observable: peak_f/spectrum
   Variables in namespace 'peak_MC':
     BackgroundRate       =       1000 │                 [fixed]                 │ Flat background rate 0
     Mu                   =       2000 │                 [fixed]                 │ Peak 0 amplitude
     E0                   =          2 │                 [fixed]                 │ Peak 0 position
     Width                =        0.5 │                 [fixed]                 │ Peak 0 width
   Variables in namespace 'peak_f':
     BackgroundRate       =       2000 │        2000±         500 [         25%] │ Flat background rate 0
     Mu                   =        100 │         100±          25 [         25%] │ Peak 0 amplitude
     E0                   =          4 │           4±         0.8 [         20%] │ Peak 0 position
     Width                =        0.2 │         0.2±        0.04 [         20%] │ Peak 0 width


Now we may plot the result with the following command line:

.. literalinclude:: ../../../macro/tutorial/fit/01_fit_script.sh
    :lines: 16-18

.. figure:: ../../img/tutorial/fit/01_fit_models_initial.png
   :align: center

   MC data defined by the model `peak_MC` and initial state of the model `peak_f`.

Next step is to define the Datasets and Analysis instances. Both of them should have unique name defined. Names should
be unique across the similar modules: all datasets should have unique names and all analysis should have unique names,
there still may exist a dataset and an analysis with the same name.

The Dataset is defined by the `dataset` module.

.. literalinclude:: ../../../macro/tutorial/fit/01_fit_script.sh
    :lines: 19

We have just created dataset `peak`, that makes a correspondence between the output `peak_f/spectrum` and output
`peak_MC/spectrum` (data). The option :code:`--asimov-data` indicates that `peak_MC` will have no fluctuations added.

In case of a single Asimov dataset the Analysis definition is straightforward:

.. literalinclude:: ../../../macro/tutorial/fit/01_fit_script.sh
    :lines: 20

Here we define Analysis, named `analysis`, which is using a single dataset `peak`. The analysis now may be used to
define the statistics. Let us use the :math:`\chi^2` statistics:

.. literalinclude:: ../../../macro/tutorial/fit/01_fit_script.sh
    :lines: 21

The syntax is similar. The module `chi2` defines the :math:`\chi^2` statistics with name `stats_chi2` and assigns it to
the analysis `analysis`.

In order to create the minimizer one needs to define its name, type, statistics and a set of parameters to minimize.
Parameters may be defined either as a list of parameter names or the namespace names. All the parameters from the
namespaces, mentioned in the command, will be used for minimization.

.. literalinclude:: ../../../macro/tutorial/fit/01_fit_script.sh
    :lines: 22

Here we define the minimizer `min`, which will use ROOT's Minuit to minimize the statistics `stats_chi2`, which depends
on the parameters, listed in the namespace `peak_f`.

The `minimizer` module only creates minimizer, but does not use it for the minimization. It may then be used from other
modules to find a best fit, estimate confidence intervals, etc. The fitting may be invoked from the `fit` module:

.. literalinclude:: ../../../macro/tutorial/fit/01_fit_script.sh
    :lines: 23

This simple module performs a fit, prints the result to standard output (:code:`-p` option) and saves it to the output
file (:code:`-o` option). After the minimization is done, the best fit values, estimated uncertainties, function at the
minimum are printed as well as some usage statistics.

.. code-block:: text
  :linenos:
  :emphasize-lines: 3,10,17

  Fit result:${
    cpu : 0.060144,
    errors : [2.10443859e+01 8.98718560e+01 1.99255272e-02 2.15730652e-02],
    errorsdict : ${
      BackgroundRate : 21.0443858746,
      Mu : 89.8718559973,
      E0 : 0.0199255272086,
      Width : 0.0215730651604,
    },
    fun : 4.51940800958e-07,
    maxcv : 0.01,
    names : ['BackgroundRate', 'Mu', 'E0', 'Width'],
    nfev : 162,
    npars : 4,
    success : True,
    wall : 0.0604510307312,
    x : [1.00000394e+03 1.99996759e+03 1.99998887e+00 4.99994251e-01],
    xdict : ${
      BackgroundRate : 1000.00394357,
      Mu : 1999.96758892,
      E0 : 1.99998886819,
      Width : 0.499994250586,
    },
  }

Extra option :code:`-s`/:code:`--set` makes fitter to set the best fit values to the model, unless the fit failed.
Now we can plot the state of the fit model.

.. literalinclude:: ../../../macro/tutorial/fit/01_fit_script.sh
    :lines: 24-

The `ns` module prints the current and default values of the parameters to the output:

.. code-block:: text

   Variables in namespace 'peak_f':
     BackgroundRate       =       1000 │        2000±         500 [         25%] │ Flat background rate 0
     Mu                   =    1999.97 │         100±          25 [         25%] │ Peak 0 amplitude
     E0                   =    1.99999 │           4±         0.8 [         20%] │ Peak 0 position
     Width                =   0.499994 │         0.2±        0.04 [         20%] │ Peak 0 width

The result of the fitting is:

.. figure:: ../../img/tutorial/fit/01_fit_models_fit.png
   :align: center

   MC data defined by the model `peak_MC` and best fit state of the model `peak_f`.

As soon as the model contains no fluctuations, the function at the minimum is consistent with zero:
:math:`\chi^2_\text{min}\approx10^{-10}`.

Also, the `fit` module have saved readable version of the result to the file :download:`01_fit_models.yaml
<../../img/tutorial/fit/01_fit_models.yaml>` in the YAML format, which can be later loaded back into python.

The full version of the command is below:

.. literalinclude:: ../../../macro/tutorial/fit/01_fit_script.sh
    :linenos:
    :caption: :download:`01_fit_script.sh <../../../macro/tutorial/fit/01_fit_script.sh>`

