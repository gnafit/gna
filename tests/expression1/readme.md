Here is a concept of the expressions in GNA.

# Some definitions

Bundle is a piece of python code that creates a small computational graph. In a lot of cases
the connections between such a graphs may be expressed in a terms of mathematical formula, or expression.

# Main objects

The main objects, used in calculations in GNA are the following:
1) output - an array or a histogram, an output of some transformation.
2) variable - usually double - a simple number. In the context of expressions the variables
   are used in the WeightedSum transformation, where the result of the transformation is
   an output S = v1*O1 + v2*O2 + ...

# The operations on the objects and their representation

Let us forget for a while who has created the variables and outputs and focus on what we can do with them.

## Variables

In terms of expression a variable is a raw variable, for example `var`. The variables may be multiplied:
`var1*var2*...` is another variable.

There is no definition of a sum of variables currently.

## Outputs

The output in terms of expression is a function call: `obj()`. This means that the `obj()` returns the output - an array
or a histogram.

A function may have some arguments or inputs. For example `fcn( obj() )` means that output of the `obj()` is connected
to the input of `fcn` object. If a function has several arguments, the inputs connected one by one. `fcn(obj1(),obj2())`
means that the the output of `obj1()` is connected to the first input of `fcn()` while the output of `obj2()` is
connected to the second input of `fcn()`.

There is a simplified version of a function call, used to make the expressions more readable that uses symbol `|`:
- `fcn| obj()` is an equivalent of `fcn( obj() )`
- `fcn1( fcn2| obj() )` is an equivalent of `fcn1| fcn2| obj()` or to the `fcn1( fcn2( obj() ) )`
- `fcn| obj1(), obj2()` is an equivalent of `fcn(obj1(), obj2())`

Literally `|` is replaced by `()` with `(` instead of `|` and with `)` at the end of the string or in a place that keeps
the brackets balanced.

The outputs may be multiplied and summed. `obj1() * obj2() * obj3()` is replaced by the output of a Product
transformation, while the `obj1() + obj2() + obj3()` is replaced by the output of a Sum transformation. The operation
priority is treated correctly in complex expressions.

**NOTE**: the objects are treated by name and the name is unique. Therefore unlike in regular programming it is
impossible to call the function twice with different arguments. The second call of the same function should be done
without arguments:
```python
fcn(obj1()) + fcn(obj2()) # Incorrect. The output of obj1() is already connected to the input of fcn()
fcn(obj1()) + fcn(obj1()) # Incorrect. The input of fcn() is already connected. No connection is needed anymore.
fcn(obj1()) + fcn()       # Correct. This is equivalent to the doubling of fcn()
```

## Outputs and variables

The main purpose of variables in expression is to weight outputs. A product of variable and output creates a WeightedSum
instance. `var * obj()` is replaced by the output of the WeightedSum() transformation.

This operation is correctly balanced with product of variables, product of outputs and sum of outputs operations.

## Constants

There is no concept of a constant currently. I.e. the code `2*var` or `2*fcn()` is not valid.

## Naming

All the objects (variables and outputs) should have unique names. Since the variables and outputs are located in the
same context, they may not have identical names. They are derived or specified by user. For example the
user may say that `o1*o2` is called `p`. Then when Expression will create the Product for `o1*o2` it will give it a name
`p`. The names mey be provided for sums and product.

# Indexing

The expressions are used with indexing in mind. A lot of parts of a model may have different versions, such as different
sources or different detectors. Consider an example of index `d` with variants `d1`, `d2`, `d3`, etc and index `s` with
variants `s1`, `s2`, `s3`, etc.

Variables and outputs may have indices. The indices are expanded for the all possible variants and combinations. Here
are the rules.

## Indexing

- `var[d]` means that there exists a variable `var` with variants `var[d1]`, `var[d2]` and `var[d3]`. It is considered
    that the variables are accessible from the current namespace by names `var.d1`, `var.d2` and `var.d3`.
- `var[d,s]` means that there exist all possible combinations of variants of `d` and `s`. The variables are located in
    a nested namespaces by names `var.d1.s1`, `var.d1.s2`, ..., `var.d2.s1`, etc. The indices ordered alphabetically.
- `obj[d,s]()` is treated the same way. The outputs are located in a nested namespace `obj.d1.s1`, etc. The inputs if
    any are also located in a nested namespace for inputs as `obj.d1.s1.00`, `obj.d1.s1.01`, etc. The numbers are used
    to represent the argument number. `fcn( obj1(), obj2() )` means that output `obj1` will be connected to input
    `fcn.00` and output `obj2` will be connected to the input `fcn.01`.

The Expression does not know how to multiplicate the object. It is considered that all the variants (of variables and
outputs) are created beforehand by bundle(s).

## Comprehension

What happens in case variables with different indices meet? Here are the rules.
- The product (or sum), means that Expression will create each possible combination:
  * `v1[s]*v2`    -> `v1.s1*v2`, `v1.s2*v2` etc.
  * `v1[s]*v2[d]` -> `v1.s1*v2.d1`, `v1.s1*v2.d2`, ..., `v1.s2*v2.d1`, etc.
  This rule is valid for sums, products and weighted sums.

- For the function call the similar rule is used:
  * `fcn[d]( obj() )` will connect the output `obj` to each input of `fcn.d1.00`, `fcn.d2.00`, etc.
  * `fcn( obj[s]() )` will connect output `obj.s1` to input `fcn.s1.00`, output `obj.s2` to input `fcn.s2.00`, etc.
    The Expression will teremine automatically, that since function's argument has indices the function output should
    also have the same indices. When Expression will call the creation of `fcn` it will require the bundle to create
    `fcn` input for each `s`.
  * The combined cases are treated properly. For `fcn[d]( obj[s]() )` the output `obj.s1` will be connected to inputs
      `fcn.d1.s1.00`, `fcn.d2.s1.00`, etc, then the output `obj.s2` will be connected to inputs `fcn.d1.s2.00`,
      `fcn.d2.s2.00`, etc, then repeat for each variant of `s`.

For a seeries of function calls the internal indices are being collected for each outer function call. For the
`outer[d]| inner2[s]| inner2[i]` the outer function will eventually have outputs with three indices `outer[d,i,s]` for
each possible combinations of `d`, `i` and `s` stored as `outer.d1.i1.s1`, etc.

## Reductions

The indices may be reduced, by applying sum and product pseudo-functions. For example:
- `sum[i]| var[i]*obj[d]()` will sum the inner expression over index `i`. The result will have index `d` remaining.
- `prod[i,d]| var[i]*obj[d]()` will multiply all the possible variants of inner expression. The result is a single
    output without indices.

Unlike functions, sum and product pseudo-functions may be called with different arguments. A sum of the outputs produces
a Sum transformation, a sum of WeighteSum outputs produces a WeightedSum output. The product of the outputs, produces
the Product transformation.

# Execution order
The following steps are used by the Expression:
1. Pre-procession. On this step the expression is converted to the valid python code. I.e. the '|' are replaced by a
   proper function calls.
2. Parsing. The expression is executed as a valid python code. For each new name, found in expression a python object is
   created. All the function calls, sums and products are represented as python objects. On this stage the Expression
   should already know what indices are available and what variants they have. Also on this stage the Expression does
   not know what is hidden behind the object names and what data is there.
3. Guessing. For each intermediate operation sum, product, indexed sum, indexed product an unique name is either derived
   or read from the configuration.
4. Building. The Expression is put into context and provided with configuration. The Expression's configuration contains
   a configuration for each primary variable and output. For each primary object name (variable or output) the
   Expression is running a bundle. The bundle is expected to provide the necessary variables, outputs and inputs for
   each combination of its indices (the relevant indices is provided to the bundle as well). When all the outputs and
   inputs are created, they are connected by the Expression.

The result of a building is a computational graph and a context. The context contains:
- `context.inputs` all the created inputs. They all should be connected.
- `context.outputs` all the created output. They all may be read.

One may print the contents of these nested dictionaries via the following python code:
```python
from gna.bindings import OutputDescriptor
print( context.inputs )
print( context.outputs )
```

The OutputDescriptor is imported for the better printing.

The variables are created in a global namespace and may be printed as well:
```python
from gna.env import env
env.globalns.printparameters()
```

#vim: textwidth=120 wrap spell

