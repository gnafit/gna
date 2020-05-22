# General notes

All users are encouranged to contribute to the project core by adding their own transformations or other code.

In order to be added to the `master`, each new tranformation/class/method should be properly documented (rst) and accompanied by one ore more dedicated test scripts.

# Commits and merging

Commits to the `master` branch are commonly prohibited. Please, contribute your code under your own branch and submit a merge request.

# Code style

* Two space indentation, no tabs.
* Class members should start with 'm_'.
* Class methods are called with explicit `this->method()` to distinguish them from static functions.
* Composite names follow camelCaseAgreement.
* TBD :)
* Use `constexpr` and `noexcept` annotations when appropriate:
  - Make sure that functions marked with `noexcept` do NOT throw any exceptions: 
    no dynamic allocations inside of functions, functions called don't throw also and etc.
  - `constexpr` everything you can

# RST documentation style

## RST sections structure

```rst
#########
Part name
#########

************
Chapter name
************

Section name
============

Subsection name
---------------

Subsubsection name
^^^^^^^^^^^^^^^^^^

Paragraph name
""""""""""""""

Subparagraph name
'''''''''''''''''

Subsubparagraph name
++++++++++++++++++++

Subsubsubparagraph name
.......................
```
