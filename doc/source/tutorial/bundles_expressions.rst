Binding graphs via mathematical expressions
"""""""""""""""""""""""""""""""""""""""""""

In the previous sections of the tutorial we have discussed how to use bundles to produces small computational graphs and
initialize parameters, as well to replicate them. The user is then left with a potentially large set of inputs, outputs
and parameters that should be bound together. Doing this manually may be quite a tedious job, which produces large
amount of hardly readable and poorly extensible code.

In the following section we will discuss a way to bind inputs, outputs and variable by means of indexed mathematical
expressions with a little amount of code.

.. contents::
    :local:

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

.. [#] Here we have to introduce a distinction: the term `Variable` relates to an instance of GNA Variable which may be
       related to the computational graph. The term `variable` relates to any variable in the code. Thus the `variable`
       may be a reference to `Variable` or to `Transformation` output. `Variable` and `Output` will sometimes be
       referred as objects.

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

Basic operations
++++++++++++++++

Within `Expression v1` the basic operations like addition and multiplication are defined. Since we deal with two kinds
of objects, `Outputs` and `Variables` the operations are defined for multiple cases:

- Product of `Variables`. The operation creates `VarProduct` object, which has a type of `Variable`.
- Product of `Variable` and `Output`. The operation creates a `WeightedSum` object instance, the result of which has a
  type of `Output`.
- Product of `Outputs`. The operation creates a `Product` object instance.
- Sum of `Outputs`. The operation creates a `Sum` object instance or a `WeightedSum` object instance.

Let look at these examples in more details.

Variables
.........

Any number of the multiplied variables will be collected into `VarProduct`:

.. code-block:: python

    a*b*c*d

Here a new variable will be created. The automatic name will be `a_times_b_times_c_times_d`. In case the library
contains a proper name and label, it will be applied.

At this moment the sum operator is not defined for the variables.

Outputs
.......

The same is applied for the `Output` objects. The product:

.. code-block:: python

    a()*b()*c()*d()

will create a `Product`. The automatic name will be `a_times_b_times_c_times_d`.

In case of the sum:

.. code-block:: python

    a()+b()+c()+d()

`Expression` will create a `Sum`. The automatic name will be `a_plus_b_plus_c_plus_d`.

Outputs and variables
.....................

When `Variable` and `Output` objects intersect, they are typically handled via `WeightedSum`:

.. code-block:: python

    a*b*c()*d()

will create a `VarProduct` instance `a_times_b`, `Product` instance `c_times_d` and `WeightedSum` instance
`a_times_b_times_c_times_d`. The `WeightedSum` handles a sum of `Outputs` with weights given by `Variables`.

When multiple sums meet together, they are combined if possible. For example:

.. code-block:: python

    a*b*c()*d() + e*f()

is again a `WeightedSum` with weights `a_times_b` and `e`, and outputs `c_times_d` and `f`.

When `WeightedSum` is multiplied by something, it extends either underlying `VarProduct` or underlying `Product`.

Indices
+++++++

The main objective of introducing of `Expressions` DSL is to provide scalability which is implemented via indexing. Any
`Variable` or `Output` may have indices assigned, which is done via square brackets. The indices are typically
initialized before the expression parsing and has the following form:

.. code-block:: python

    nidx = [
        ('d', 'detector',    ['D1', 'D2', 'D3']),
        ['r', 'reactor',     ['R1', 'R2', 'R3']]
    ]

The syntax for each entry is `(index, label, variants)`. The index is a variable name, that may be used within
expression:

.. code-block:: python

    a[d]

will request a bundle to build variable `a`. It will also pass a list of index values `['D1', 'D2']` so the bundle has
to provide a `Variable` instance for each of the indices. When multiple indices occur:

.. code-block:: python

    a[d,r]()

the bundle will be requested to provide and instance (output in this case) for each of the combinations of values of
indices `d` and `r`.

.. note::

    The indices has to be specified only for the first object usage. In the subsequent cases the indices may be omitted.
    It is better to still write them explicitly to keep track. An exception will be raised in case inconsistent indices
    are used.

.. note::

    An absence of explicit index specification implies that the object has *empty* index with a single value.

Index values combinations
.........................

In most of operations: sums, products, function calls the indices are collected. The `Expressions` mechanism handles all
the combinations of collected indices. For example:

.. code-block:: python

    a[i]*b[j]() + c[l]()

request bundles to create objects `a`, `b` and `c`. Each of them has its own index. The `WeightedSum` instance
`a_times_b` will be created for each combination of values of indices `i` and `j`. The final `Sum` instance will be
created for each combination of values of `i`, `j` and `l`.

Using this approach enables the user, for example, to disable most of the detectors simply by specifying a truncated
list of values for index `d`.

The function call (connecting outputs to inputs) is handled in a same fashion. The code:

.. code-block:: python

    fcn[l]( a[i]*b[j]() + c[l]() )

collects the indices for the `fcn` `Output`/`Input` pair as well. The `fcn` object has a single explicit index `l`. Its
argument has a set of three indices `i`, `j` and `l`. Therefore the bundle, which creates `fcn` will be requested to
make an `fcn` input and `fcn` output for each combination of values of indices `i`, `j` and `l`.

Reductions
++++++++++

The number of object instances may be reduced via `sum`, `prod` or `concat` functions. The code

.. code-block:: python

    sum[j]| fcn[l]( a[i]*b[j]() + c[l]() )

will make a `Sum` or `WeightedSum` instance for each combination of `i` and `l` indices, summing together outputs for
all the possible values of `j`. Similarly

.. code-block:: python

    prod[l]| sum[j]| fcn[l]( a[i]*b[j]() + c[l]() )

will make a `Product` instance for each value of index `i` by multiplying together outputs for all the possible values
of `j`. The resulting object will have an output for each value of index `i`.

The `concat` function works the same way and does the concatenation.

Name guessing
+++++++++++++

Storage
+++++++

Examples
''''''''
