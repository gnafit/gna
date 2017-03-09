Example
===========
Let's try to do some example from
scratch.

Let's assume we have a very simple observable of event counts N:
constant background :math:`b` plus signal with strength :math:`\mu`,
which is Gaussian-shaped peak at :math:`E_0` with width
:math:`w`. This is just random example without any physical
background. The formula is   

.. math::
   \frac{d N}{d E} = b + \mu \frac{1}{\sqrt{2\pi w}}\exp{\frac{-(E-E_0)}{2w^2}}

How can we implement this in the code? There are actually different
ways, it's up to you which one to chose. The most simple and concise,
is to implement all the calculations in single ``GNAObject``:

.. literalinclude:: examples/GaussianPeakWithBackground.hh
   :language: c++
   :linenos:

Everything is in one header file, just for consciousness. The code
itself is hopefully understandable, but anyway, let's go line by line.

All the computation code goes to one class that should be derived from
GNAObject. First of all, you'll need to include ``GNAObject.hh`` if
you want to use it. You also need to derive from
``Transformation<your-class-name>`` to make things work properly.

In our class constructor, we define the variables which we are going
to use in our computations. The syntax is simple:
``variable_(pointer-to-variable, name)``. The actual variables may be
actually stored anywhere, but they shouldn't be shared across
different instances, so it's natural to make them class members. We
have four variables, and all of them are defined.

Then we need to define our transformation. The definition is more
complicated, so the syntax. The lines 18-23 are actually one
statement of chained member calls, separated by newline for clarity.
The first line is simple, just ``transformation_(this, name)``. Always
pass ``this`` as the first argument. Then input definition goes, it's
just ``.input(name)``, there may be any number of them, but the order
is important, as you'll see later. The same with the outputs.

Then the *types function* comes with such syntax: ``.types(func1[, func2,
...)``. All the functions should check if the actual input types are
acceptable and construct the output types based on them. You can
provide any class member function with signature ``void T::func(Atypes
args, Rtypes rets)`` or usual ``std::function`` (for example, a
lambda) with signature ``void func(T obj, Atypes args, Rtypes
rets)``, where ``T`` is your class name. Here we use the predefined
``Atypes::pass<I,J>`` function which just assigns the type of ``I``-th
input to  ``J``-th output (check that in ``core/TransformationBase.hh``).

Finally, the *function* which does all the calculation is specified
with ``.func(func)``, there can be only one of them and it should have
have signature ``void T::func(Args args, Rets rets)`` or ``void
func(T obj, Rets args, Rets rets)``. Here we used the first
possibility and as you can see in the first line, the ``calcRate``
member function does really implement our formula. A few notes on
``Args`` and ``Rets``. They both provide indexed (zero-based) access
to the data of corresponding inputs and outputs, in the same order as
in the transformation definition. We have only one input, and one
output, hence ``args[0]`` and ``rets[0]`` make sense. 

Each object returned by ``Args`` or ``Rets`` is a ``Data`` object: a
shapeless buffer of ``double`` plus ``DataType`` type descriptor
(check ``core/Data.hh``). It's up to you how to deal with that data,
you can either read/write to the buffer directly or use convenient
interface provided by the ``Eigen`` library and treat the buffer like one
or two dimensional array with coefficient-wise operations in
``numpy``-style (``Data::arr`` and ``Data::arr2d`` objects), or treat them
like vector or matrix and use linear algebra operations (``Data::vec``
and ``Data::mat``). It's often convenient to make aliases with the
corresponding references, as we did with ``E``.  There are almost no
shape checks in runtime, so **please check everything** carefully in the
types function. As for ``variable<double>``, you can treat it just
like ``double`` in most cases. If implicit unboxing is not possible,
you can always do it with ``variable<double>::value()``.

To make that code usable you need:

- put it somewhere in the source tree (for example, let's create
  ``examples/`` directory, and put the code in the
  ``examples/GaussianPeakWithBackground.hh`` file);
- modify the ``CMakeLists.txt``: if we would have any C++ source
  files, we would add it to ``SOURCES`` list; since we have only
  header, we add only header to the ``HEADERS`` list;
- add new directory ``examples/`` to the ``include_directories``;
- modify `core/LinkDef.h` adding the following line::
    #pragma link C++ class GaussianPeakWithBackground-;

- recompile: ``make -C build``.

To actually use the computational module we have to create a Python
initialization code. Here is a possible way to do it:

.. literalinclude:: examples/gaussianpeak.py
   :language: py
   :linenos:

First of all, there are few imports: we need ``basecmd`` to make code
executable, as the ``dispatch.py`` requires; ``env`` to have access to
the shared state (we need it for example to handle parameters, which
may be shared between different models); ``ROOT`` to have all our
compiled parts accessible and ``np`` for the initialization-time
computations.

We create class ``cmd`` derived from ``basecmd``. The ``cmd`` name is
fixed and should be used in each ``gna`` module. The common way to
define arguments for our module is to use the ``initparser`` class
method. It passes the ``ArgumentParser`` object of the standard
``argparse`` as ``parser`` and an ``env`` object. It's actually the
same as global ``env`` imported from (``gna.env``) and may be
ignored. Five arguments are defined:

- ``--name`` to provide an unique identifier for the theoretical prediction results 
  that will be initialized in the module, it is  used later in other
  modules to get access to the prediction;
- ``--Emin``, ``--Emax`` and ``--bins`` to specify the binning
  properties of output histogram;
- ``--order`` to specify integration order. Gauss-Legendre algorithm
  is used to integrate rate into bins of the histogram and the
  specified number of points will be used in each bin.

The actual code to initialize the prediction is in the ``init``
method. The result of arguments parsing (again, just standard
``argparse``) is in ``self.opts``. There we initialize a *namespace*
(commonly abbreviated as ns) with the name provided by
``--name``. All the experiment-specific parameters and outputs should
be placed there. The parameters are initialized in lines 17-20, and
``ns.reqparameter`` method is used. This method searches for already
defined parameter with the specified name in currently available
namespaces, and in case it isn't found, creates it in the namespace
``ns`` according to the provided ``kwargs``. So, if some of that
parameters were already created by other module and the corresponding
namespace is explicitly activated, those parameters will be
reused. This makes possible to share parameters between different
experiments.

Then we activate the created namespace (line 21) for a moment to
create our ``GaussianPeakWithBackground`` computational block. All 
variables in it will be bound, and the previous ``reqparameter`` calls
ensure that all the required names will be available during the
binding procedure.

In lines 24-25 the edges of the histogram are computed, and array of
integration orders is filled. We use the same order for all bins for
simplicity.

To actually fill the histogram, two more computational blocks are
required. They all correspond to integration process. As for now, only
Gauss-Legendre integration is implemented, so we'll use
``GaussLegendre`` and ``GaussLegendreHist`` objects -- the former
provides points where the integrand should be computed, and sums the
result with the corresponding Gauss-Legendre weights. They are also
implemented in terms of transformations, so ``GaussLegendre`` provides
transformation ``points`` with no inputs and one output ``x``, while
``GaussLegendreHist`` provides transformation ``hist`` with one input
``f`` and output ``hist``. To access them, the following syntax is
used:

- ``object.name`` returns transformation ``name`` of
  object. Alternatively, ``object.transformations`` dict-like object
  may be used, for example, to iterate the transformation;
- ``transformation.name`` will give access to input or output named
  ``name`` of the transformation. If there is input and output with
  the same name, it will raise an exception. In this case more
  qualified access is required: ``transformation.inputs.name`` or
  ``transformation.outputs.name``. Again, ``inputs`` and ``outputs``
  are dict-like objects and may be used for iteration.

It worth noting, that the mentioned dict-like objects allow also
index-based and attribute-based access.

Those accessors are generally used to connect input and outputs:

- the basic syntax is just ``__call__``: ``inp(out)`` will make
  ``out`` source of values for ``inp``. Alternatively, you can write
  ``inp.connect(out)``.
- as a shorthand, as ``out`` you can pas not only object returned by
  ``obj.transformation.outputs.name`` but any single-outputed object:
  ``obj.transformation.outputs`` or ``obj.transformation`` if the
  transformation has only one output, or even ``obj`` itself it is
  explicitly derived from ``GNASingleObject`` and its transformation
  has only one output.
- another shorthand is to connect all inputs of ``transformation1`` to
  outputs of ``transformation2``:
  ``transformation1.inputs(transformation2)``. The count of
  inputs/outputs should be the same. Only order is important, no name
  checking is performed.

That's what is used in lines 28 and 30 -- points from integrator are
passed to the rate calculator (output ``integrator.points.x`` is
connected to input ``model.rate.E``), and the calculated rate is
passed for summation into histogram (output ``model.rate.rate`` is
connected to input ``hist.hist.f``).

The final output of our computations is stored into namespace as an
*observable*. The function ``ns.addobservable`` will check, if there
is no free inputs in the whole subgraph, leading to the provided
output and will make it accessible for future use under the given name
inside the namespace ``ns``.


As an example try to implement the exponential background distribution with a
Gaussian peak.
