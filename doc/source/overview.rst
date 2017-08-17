Overview
=====================

The codebase consists of two parts. The code where computations are
implemented is written in C++11 and is scattered in different
top-level directories

.. code-block:: bash

  core/
  neutrino/
  integrator/
  detector/

The paths to compiled source files should be explicetely listed in the
``CMakeLists.txt``.

Computation flow configuration, user interaction and everything else
is written in Python 2 and placed in the ``pylib/`` directory.

The two parts are glued together with PyROOT. Although saving some
coding, PyROOT is slow and too implicit. I would seriously consider
dropping it in favor of something more cleaner (maybe `pybind11
<https://github.com/wjakob/pybind11/>`_) in the future.

C++ side
************************
All the objects, which are going to be used in the computations are
derived from the ``GNAObject`` class. This object provides the
following basic features.

Variables, parameters, evaluables and taintflags
###################################################
A *variable* represents a named value, which is going to be changed
during fit procedure; it's generally a parameter on which the
theoretical prediction of experemental observations depends. Variables
are implemented as template class ``variable<T>`` (where ``double`` is
almost the only tested and used choice for ``T``) and provide the
status tracking: when some variable ``v`` is modified, any other
variable (or a similar object described below) ``x`` which is declared
to depend on ``v`` will be signalled to invalidate currently computed
value indicating for later recomputation. The invalidation procedure
is internally called *tainting*, while dependencies are named
*subscriptions*: ``v.subscribe(x)`` literally means that ``x``
subscribes to invalidation events of ``v``.

The users of variables are not expected to change their values, only
to use them in computations, so the ``variable<T>`` class does not
have any direct modifications method. To make it actually usable (to
make them hold some useful value), it should by associated to either
*parameter* (implemented as ``parameter<T>`` or *evaluable*
(``evaluable<T>``).  The first one provides an interface to directly
modify the value (``set()`` method), while the second is initialized
with a function to compute the value, usually depending on some other
variables. The procedure of such association is called *binding* and
is done almost exclusively on the Python side of the code. All the
variables of constructed objects are assumed (except explicetely
stated on Python side during the creation) to be bound to
something. Any number of variables of different objects may be bound
to the same parameter or evaluable, making it possible to have common
systematics. The computation code should not care about bindings
themselves, no matter whether it will be an independend parameter, or
dependant evaluable, the code should just use the ``variable<T>``
interface.

A *taintflag* is an object used to expose the invalidation state
outside of the parameter/variable/evaluable. It may be subscribed to
a number of trackable objects and casting to bool will tell you,
whether any of them was changed since last reset.

Please refer to the code in ``core/Parameters.hh`` where the
``variable<T>``, ``parameter<T>``, ``evaluable<T>`` and ``taintflag``
machinery is implemented. They are exposed in ``GNAObject`` through
the ``Parametrized`` base class, which is implemented in
``core/Parametrized.hh``.

Transformations
#################
A *transformation* represents the
actual computation procedure -- it's basically a function taking any
number of inputs (or sometimes internally called *sources*) and
providing arbitrary number of outputs (or internally *sinks*). The
number, size and all other properties of inputs and outputs are
determined during the initialization stage, starting from the
object creation and ending at the first request of actual computation
results. The function that determines outputs properties (like sizes)
from the inputs properties is called *types function* and it's
expected to be called only during initialization stage, while the computation
itself is done by some *function* which is called only after the types
function successfully precomputed everything possible.

To carry out the computations there should be no free inputs, all of
them should be connected to output of another object, forming the
computation chain, computation graph (which is acyclic and directed).
The transformation gets invalidated when any of the transformations
connected to it is invalidated or when a variable of the containing
``GNAObject`` gets invalidated (unless another behaviour is explicetely
specified during the transformation initialization). The computed results are
reused making partial computation faster as long as it stays valid.

There may be any number of transformation provided by a single
``GNAObject``, they all will have its state as shared. If only one
transformation is expected, one can use the ``GNASingleObject``
instead, which allows to drop the name of transformation in some
contexts making code shorter.

The transformations machinery is implemented by inheriting from two
classes: ``TransformationTypes::TransformationBase``, which is
generally inherited only through ``GNAObject`` and should not be used
directly, and ``Transformation<T>`` template class which should be
inherited by each user of transformations with the class itself as
``T`` (CRTP is employed).

The code can be found in ``core/TransformationBase.hh`` and
``core/Transformation.hh``. The former is (at least was) intended to be
used only by C++, while the latter to be exported to Python side with
PyROOT. This was done with ROOT5 in mind, since ROOT6 is very good at
C++11 bindings, the distinction is not as important anymore.

Python side
****************
The entry point of python code is ``run()`` function from ``gna.dispatch``. It
handles command line arguments  parsing and runs specified commands
from ``gna.ui``. Each command corresponds to one module in ``gna.ui``,
it should contain ``cmd`` class which should be derived at least from
``basecmd``. Methods ``init()`` and ``run()`` are executed, the former
exactly after the latter, so there is no real difference between
them. Commands are executed strictly sequentially in the order
specified in the command line.

Commands do everything -- define experiments, data, do fits, plotting,
etc. They share common state with ``env`` object. This
object is implemented in ``gna.env`` and is assumed to be constructed
only once in the whole program, the instance is available as
``gna.gna.env``. Sometimes you can see the env object passed around as
it wasn't single, this behaviour is old (before there were plans to
have multiple ``env`` objects) and considered deprecated.

Everytime ``GNAObject`` is created, it's registered in the
``env``. This means (at least):

- a reference to the object is kept somewhere in the env, in order not to
  bother with memory management (ffffuuuuuuuuuu)
- each variable of the object is bound to some value (parameter or
  evaluable); if a variable can't be bound and isn't declared optional
  by the object itself, an exception is thrown.
- each evaluable expression is registered by the corresponding name in the
  namespace provided with ``ns`` kwarg to constructor.

This is implemented in ``gna.bindings``, where some pythonization and
monkeypatching is done.

The parameters and evaluables are collected into hierarchical
namespaces. Their names inside a namespace are unique, but they can
coincide with subnamespace names (not recommended though to avoid
confusion). By convention, parameters and evaluables names are written
with initial uppar case, while namespaces are lower case.

Some namespaces may be active, so names inside them will be visible
with ``env.pars`` and will be available during binding
procude. Inactive namespaces are invisible until activated. This is
handled by the ``nsview`` object inside
``env``. Activation/deactivation is done with the context syntax
(``with ns: ...``) or with explicit
``ns.add([...])`` / ``ns.remove([...])``.
