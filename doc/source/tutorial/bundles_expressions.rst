Binding graphs via mathematical expressions
"""""""""""""""""""""""""""""""""""""""""""

In the previous sections of the tutorial we have discussed how to use bundles to produces small computational graphs and
initialize parameters, as well to replicate them. The user is then left with a potentially large set of inputs, outputs
and parameters that should be bound together. Doing this manually may be quite a tedious job, which produces large
amount of hardly readable and poorly extensible code.

In the following section we will discuss a way to bind inputs, outputs and variable by means of indexed mathematical
expressions with a little amount of code.

Design principles
'''''''''''''''''

The GNA `expressions` is a Domain Specific Language that has the following goals:

- Describe the computational graph structure in a concise and scalable way.
- Automatically create intermediate transformations and variables, derive their names.
- Execute bundles to provide necessary constituents (inputs, outputs).
- To create an expression a set of 3 elements should be provided:

  + The expression itself.
  + A dictionary with configurations for bundles.
  + A dictionary with rules to derive intermediate names and labels.

By using the scheme, described above we ensure that:

- The configuration is detached from the code:

    + The configuration is specified for each particular bundle. It is detached from all other bundles and from the
      `Expression` code.
    + It is not necessary to declare intermediate objects. The intermediate objects are created automatically. The names
      are either assigned automatically or derived based on the naming configuration, which is provided independently.
- Thus the expression code, which defines the connections may be written in a concise manner, not interrupted by
  the configuration lines, introducing intermediate objects, etc.
- The computational graph setup is done blindly, without an access to the actual data or even the data type and shape.
  The consistency check is then done by the GNA core.

The current `expressions` implementation is highly experimental and raw and due to be updated in the future. Since the
design of the `expressions` may be very specific to the particular task the goal is to have multiple implementations of
the `expressions` language, that may be interchanged.

In the following text we describe the implementation `Expressions v1`.

Expressions v1
''''''''''''''

Main building blocks
++++++++++++++++++++

Its basic structure represents the objects that are available in GNA: `Variables` and `Transformations`. The code itself
is almost valid Python code with the only extension — right associated function call operator `|`.

Within expression code it assumed that any variable [#]_ exists if it is used. Its type is derived from its usage. The
minimal code is

.. code-block:: python

    a

in which we retrieve a `Variable` with name `a`. As the variable is triggered, the `Expression` will loop over the list
of configurations to find the particular pair (bundle, configuration) that may provide the variable a. In case the pair
is found the bundle will be executed. In case the pair is not found, an exception will be issued.

The output of a transformation in `Expressions v1` is represented by a function call. The code

.. code-block:: python

    a()

will work the same way as the previous one with the only difference: the `Expression` will search for a (bundle,
configuration) that is able to provide an Transformation output, named `a`. An empty argument lists indicates that an
object `a` has an output and no inputs.

The function call represents a binding action, when the input is connected to the output. The code

.. code-block:: python

    b(a())

will require bundles to create outputs `a` and `b` and an input for `b`. The output `a` is then connected to the input
`b`.

.. caution::

    Once the connection is performed it may not be changed. Any subsequent useage of the output `b()` should be used
    with an empty argument list.

Multiple arguments are supported. The code

.. code-block:: python

    b(a(), c())

operates with outputs `a()` and `c()`, which are connected to the two inputs of `b`. After execution all the inputs and
outputs, used in the expression are available.

Function call operator |
++++++++++++++++++++++++

Long expressions may contain lots of function calls, which is difficult to read. In order to improve the readability we
introduce a special operator `|`, which represents a function call. Therefore the code:

.. code-block:: python

    a(b(c(d(e(f())))))

may be rewritten as:

.. code-block:: python

    a| b| c| d| e| f()


In the `Expression v1` this is implemented as a character substitution. Each occurrence of `|` is replaced by `(`, an
extra `)` is added in a balanced way. The `|` operator is right-to-left associativity, like power operator `**` and
unlike actual bitwise OR operator `|`, which is not used in this context.

The combined syntax is treated as follows. The code

.. code-block:: python

    a(b| c(), d())

is **equivalent** to

.. code-block:: python

    a(b(c(), d()))

and **not equivalent** to:

.. code-block:: python

    a(b(c()), d())

.. [#] Here we have to introduce a distinction: the term `Variable` relates to an instance of GNA Variable which may be
       related to the computational graph. The term `variable` relates to any variable in the code. Thus the `variable`
       may be a reference to `Variable` or to `Transformation` output.

