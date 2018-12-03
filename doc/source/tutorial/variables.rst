Variables (TBD)
^^^^^^^^^^^^^^^

Combining objects that do not change is not very interesting task. The GNA was designed in order to work with models
with large number of parameters. In the simplest case the variables represent weights in a sum of arrays
(:ref:`WeightedSum`). In the more complex case variables represent some physical parameters, like parameters of energy
resolution model (:ref:`EnergyResolution`).

We will start working with variables by introducing the concept of environment and namespaces.

Environment
"""""""""""

Environment is a global GNA object defined in Python and holding a folder like structure with variables.
When ``GNAObject`` is created it requests a list of variables it depends on from the environment. The variables are then
bound to the transformations. As the value of the variable changed the `taintflag=false` is propagated to the dependent
variables and transformations invalidating the cached data.

Defining parameters
"""""""""""""""""""

Gaussian parameters
'''''''''''''''''''

Since it is typically required to propagate the systematic uncertainty it is implied that each parameter is associated
with some probability distribution. By default it is normal distribution with central value and sigma.

Let us look at the following example:

.. literalinclude:: ../../../macro/tutorial/variables/01_variables.py
    :linenos:
    :lines: 4-
    :caption: :download:`01_variables.py <../../../macro/tutorial/variables/01_variables.py>`

The parameters are created within namespaces via method ``defparameter(name,...)``. The main namespace is
``env.globalns``. We create parameter `par_constrained` with central value of 1.0 and absolute uncertainty of 0.1. The
optional argument `label` defines text, which will be printed in the parameter listing:

.. literalinclude:: ../../../macro/tutorial/variables/01_variables.py
    :lines: 12

Instead of `sigma` one may use keyword `relsigma` in order to provide the relative uncertainty:

.. literalinclude:: ../../../macro/tutorial/variables/01_variables.py
    :lines: 13

In case the minimization with nuisance parameters is requested the uncertainty value is used to setup the nuisance term
for the parameters.

The parameter may be fixed. In this case it will not be passed to the minimizer or derivative calculation.

.. literalinclude:: ../../../macro/tutorial/variables/01_variables.py
    :lines: 14

The parameter may be set free. In this case the nuisance parameter *is not* added to the likelihood function and the
parameter value will be fitted without constraints.

.. literalinclude:: ../../../macro/tutorial/variables/01_variables.py
    :lines: 15

In addition to the ``defparameter(name,...)`` another method ``reqparameter(name,...)`` is defined. The latter one does
not create a new parameter in case the parameter with the same name is already defined. Using ``reqparameter`` in the code
enables the possibility to predefine the parameter before the code is executed. Thus the line

.. literalinclude:: ../../../macro/tutorial/variables/01_variables.py
    :lines: 16

will do nothing since the parameter `par_free` already exists.

Finally the parameter listing may be printed to the terminal:

.. literalinclude:: ../../../macro/tutorial/variables/01_variables.py
    :lines: 19

The command produces the following (colored) output:

.. code-block:: text

   Variables in namespace '':
   par_constrained      =         10 │          10±         0.1 [          1%] │ Constrained parameter (absolute)
   par_constrained_rel  =         10 │          10±           1 [         10%] │ Constrained parameter (relative)
   par_fixed            =          1 │                 [fixed]                 │ Fixed parameter
   par_free             =          1 │           1±         inf [free]         │ Free parameter

The printout contains the following columns:

    1. Parameter name.
    2. Symbol **=**.
    3. Current value of the parameter.
    4. Symbol **|**.
    5. Central/default value of the parameter or the flag `[fixed]`.
    6. Uncertainty of the parameter.
    7. *(optional)* Depending on the type and state of the parameter:

        - Relative uncertainty in case central value is not 0.
        - *Current* value of the angle in number of :math:`\pi`.
        - A flag `[free]` in case parameter is free.
    8. Symbol **|**.
    9. *(optional)* Parameter limits.
    10. Symbol **|**.
    11. *(optional)* Parameter description.

Labels are printed only in case `labels=True` argument is specified.

.. hint::

    It's highly recommended to add the proper description (including units) to the parameters from the very beginning.

Nested namespaces
'''''''''''''''''

The environment has directory-like structure: a group of parameters may be defined as nested namespace (without depth
limit). A nested namespace may be created by calling a namespace and passing new namespace name as an argument:

.. literalinclude:: ../../../macro/tutorial/variables/02_nested_variables.py
    :linenos:
    :lines: 4-
    :emphasize-lines: 12,19
    :caption: :download:`02_nested_variables.py <../../../macro/tutorial/variables/02_nested_variables.py>`

Here the nested namespace `eff` is created via call:

.. literalinclude:: ../../../macro/tutorial/variables/02_nested_variables.py
    :lines: 15

The variable `effns` is then used just like `env.globalns` to create new variables:

.. literalinclude:: ../../../macro/tutorial/variables/02_nested_variables.py
    :lines: 18-19

There is no need to refer to the nested namespace directly. Alternatively the namespace name(s) may be added in front of
variable name with `.` (period) used to split names:

.. literalinclude:: ../../../macro/tutorial/variables/02_nested_variables.py
    :lines: 22

Here `eff.eff3` means variable `eff3` in namespace `eff`. The following lines are used to demonstrate the ability to
create the deeper set of the parameters:

.. literalinclude:: ../../../macro/tutorial/variables/02_nested_variables.py
    :lines: 25-26

The example produces the following output:

.. code-block:: text

    Variables in namespace '':
      norm                 =          1 │           1±         0.1 [         10%] │ Normalization
    Variables in namespace 'eff':
      eff1                 =        0.9 │         0.9±        0.01 [    1.11111%] │ Efficiency #1
      eff2                 =        0.8 │         0.8±        0.02 [        2.5%] │ Efficiency #2
      eff3                 =       0.95 │        0.95±       0.015 [    1.57895%] │ Efficiency #3
    Variables in namespace 'misc.pars':
      par1                 =         10 │          10±           1 [         10%] │ Some parameter
      par2                 =         11 │          11±           1 [    9.09091%] │ One more parameter

The nested namespaces and parameters are printed on the same level to keep more space for the details.

Changing the value of the parameters and working with angles
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

The following example introduces the uniform angle parameter and shows some examples of how to get and set value for it
and normal parameters.

.. literalinclude:: ../../../macro/tutorial/variables/03_angle_getset.py
    :linenos:
    :lines: 4-
    :emphasize-lines: 11,20,21,26,32,33
    :caption: :download:`03_angle_getset.py <../../../macro/tutorial/variables/03_angle_getset.py>`

The uniform angle parameter may be defined by the ``defparameter()``/``reqparameter()`` methods by passing
`type='uniformangle'` option:

.. literalinclude:: ../../../macro/tutorial/variables/03_angle_getset.py
    :lines: 14

The main feature of uniform angle is that its value is converted to the limits of :math:`(-\pi,\pi)`. This enables
proper minimization with periodicity taken into account.

In the example we have defined two parameters:

.. code-block:: text

   Variables in namespace '':
     norm                 =          1 │           1±         0.1 [         10%] │ Normalization
     phase                =     1.5708 │      1.5708              [        0.5π] │  (-π, π)                    │ Phase angle

The parameter instances may be obtained by assigning the result of the ``defparameter()``/``reqparameter()`` call

.. literalinclude:: ../../../macro/tutorial/variables/03_angle_getset.py
    :lines: 13

or requested by the namespace field access:

.. literalinclude:: ../../../macro/tutorial/variables/03_angle_getset.py
    :lines: 16

Then the ``set(value)`` method may be used to change the current value of the parameter:

.. literalinclude:: ../../../macro/tutorial/variables/03_angle_getset.py
    :lines: 23-24

This result may be printed to the terminal:

.. code-block:: text

   Variables in namespace '':
     norm                 =          2 │           1±         0.1 [         10%] │ Normalization
     phase                =-0.00318531 │      1.5708              [-0.00101391π] │  (-π, π)                    │ Phase angle

Note, that the value slightly below than :math:`2\pi` was converted to the value close to 0 for the angle.

There is an extra method that enables the user to change the value of the normally distributed variable in a normalized
way, in other words the use may set the number of sigmas the value deviates from the central value with the following
command:

.. literalinclude:: ../../../macro/tutorial/variables/03_angle_getset.py
    :lines: 29

Setting the deviation by :math:`1\sigma` produces the following result.

.. code-block:: text

    norm                 =        1.1 │           1±         0.1 [         10%] │ Normalization

Finally, the values of the variables may be read back by the ``value()`` method:

.. literalinclude:: ../../../macro/tutorial/variables/03_angle_getset.py
    :lines: 35-38

Here is the output of the code:

.. code-block:: text

   Norm 1.1
   Phase -3.13

WeightedSum transformation
""""""""""""""""""""""""""

After getting familiar with how variables are defined it is the time to start using them with transformations. The first
transformation we will consider is :ref:`WeightedSum`. ``WeightedSum`` transformation implements the following sum:

.. math::
    S = \sum_{i=1}^{N} \omega_i A_i,

where :math:`A_i` is an array bound to the transformation input :math:`i`, :math:`\omega_i` is a weight bound to the
transformation from the namespace. Let us look at the example:

.. literalinclude:: ../../../macro/tutorial/variables/04_weightedsum.py
    :linenos:
    :lines: 4-64, 70
    :emphasize-lines: 30
    :caption: :download:`04_weightedsum.py <../../../macro/tutorial/variables/04_weightedsum.py>`

The example implements the following formula:

.. math::

   S = a\sin(x)+b\sin(16x)+c\cos(16x),

where :math:`a`, :math:`b` and :math:`c` are variables, initialized as follows:

.. literalinclude:: ../../../macro/tutorial/variables/04_weightedsum.py
    :lines: 16-18

The ``WeightedSum`` constructor is similar to the one for ``Sum`` with the only addition: a list of variables names
should be passed as the first argument:

.. literalinclude:: ../../../macro/tutorial/variables/04_weightedsum.py
    :lines: 32-33

The second argument is either list of outputs or list of names to give to the inputs. In the latter case the inputs
should be bounded to outputs of other transformations manually as it was done in tutorials for transformations
:ref:`tutorial_sum` and :ref:`tutorial_product`.

The contents of the ``WeightedSum`` from the example above is the following:

.. code-block:: text

   [obj] WeightedSum: 1 transformation(s)
        0 [trans] sum: 3 input(s), 1 output(s)
            0 [in]  a -> [out] points: array 1d, shape 500, size 500
            1 [in]  b -> [out] points: array 1d, shape 500, size 500
            2 [in]  c -> [out] points: array 1d, shape 500, size 500
            0 [out] sum: array 1d, shape 500, size 500

After initializing the object we make four plots:

.. figure:: ../../img/tutorial/04_weightedsum.png
    :align: center

    A set of ``WeightedSum`` plots for different values of the parameters.

First (blue) line corresponds to the initial parameters :math:`a=1.0`, :math:`b=0.1` and :math:`c=0.05`. Then we switch
of fast oscillating components with

.. literalinclude:: ../../../macro/tutorial/variables/04_weightedsum.py
    :lines: 48-49

Here we have used `push(value)` method, that sets the value of the parameter and saves the previous one. The previous
value may then be retrieved by `pop()` method.

The second line (orange) corresponds to the main frequency. Then we disable main frequency and enable the first
secondary frequency by:

.. literalinclude:: ../../../macro/tutorial/variables/04_weightedsum.py
    :lines: 53-54

The green line corresponds to this. The last minor component (red) is plotted after changing the parameters:

.. literalinclude:: ../../../macro/tutorial/variables/04_weightedsum.py
    :lines: 58-59
