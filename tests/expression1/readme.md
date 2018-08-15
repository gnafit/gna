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

## Variables

In terms of expression a variable is a raw variable, for example 'var'. The variables may be multiplied:
'var1*var2*...' is another variable.

There is not definition of a sum of variables currently.

## Outputs

The output in terms of expression is a function call: 'obj()'. This means that the 'obj()' returns the output - an array
or a histogram.

A function may have some arguments or inputs. For example 'fcn( obj() )' means that output of the 'obj()' is connected
to the input of 'fcn' object. If a function has several arguments, the inputs connected one by one. 'fcn(obj1(),obj2())'
means that the the output of 'obj1()' is connected to the first input of 'fcn()' while the output of 'obj2()' is
connected to the second input of 'fcn()'.

There is a simplified version of a function call, used to make the expressions more readable that uses symbol '|':
- 'fcn| obj()' is an equivalent of 'fcn( obj() )'
- 'fcn1( fcn2| obj() )' is an equivalent of 'fcn1| fcn2| obj()' or to the 'fcn1( fcn2( obj() ) )'
- 'fcn| obj1(), obj2()' is an equivalent of 'fcn(obj1(), obj2())'

Literally '|' is replaced by '()' with '(' instead of '|' and with ')' at the end of the string or in a place that keeps
the brackets balanced.

The outputs may be multiplied and summed. 'obj1() * obj2() * obj3()' is replaced by the output of a Product
transformation, while the 'obj1() + obj2() + obj3()' is replaced by the output of a Sum transformation. The operation
priority is treated correctly in complex expressions.

## Outputs and variables

The main purpose of variables in expression is to weight outputs. A product of variable and output creates a WeightedSum
instance. 'var * obj()' is replaced by the output of the WeightedSum() transformation.

This operation is correctly balanced with product of variables, product of outputs and sum of outputs operations.

# Indexing

The expressions are used with indexing in mind. A lot of parts of a model may have different versions, such as different
sources or different detectors. Consider an example of index 'd' with variants 'd1', 'd2', 'd3', etc and index 's' with
variants 's1', s2', s3', etc.

# vim: textwidth=120 wrap

