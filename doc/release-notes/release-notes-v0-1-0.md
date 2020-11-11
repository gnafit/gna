# Release notes for GNA v0.1.0

This is a first tagged version of GNA. The major updates include:

[[_TOC_]]

## Stack update

### C++17 support

GNA is now using some C++17 features and is compiled with C++17 support and requires `gcc>=7.0` or `clang>=4.0`. The ROOT should be compiled with C++17 support as well.

### Python3 migration

GNA now fully supports Python3. While, the performance is tested on sum fits, some shortcomings may be occasionally observed. The Update requires ROOT>=6.22 compiled with Python3 support.

### Summary

New version requires ROOT>=6.22, compiled with Python3 and C++17 support. Example ROOT configuration reads as follows:

```sh
cmake ../root-6.22.02 -D CMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=/data/work/soft/root-6.18.04_install_p2_gcc17 -Dgnuinstall:bool=ON -Dminuit2:bool=ON
```

## Modularity and package organization

GNA now provides a convenient way to manage closely related or user generated GNA parts as packages. Currently the package may contain:

* UI modules 
* Bundles
* Experiments
* Python libraries

The packages may be stored in a sub folders of:

* Default search path `package/`
* Directories, listed in environment variable `$GNAPATH`

For each subdirectory of `package/` or `$GNAPATH/` may contain folders `ui/`, `bundles/`, `experiments/` and `lib/` to search for the items respectively.
While UI modules, bundles and experiments are searched automatically in paths `package/*/ui`, `package/*/bundles` and `package/*/experiments` (as well as in paths from `$GNAPATH`), the libraries should be imported by their full module path.

## Storage update

### Introduction

With this release we start migrating to the new internal storage format. It is indended to be as simple and consistent as possible: a set of nested dictionaries. In order to make the handling easier we introduce a special wrapper `DictWrapper`. Its purpose is only to wrap a set of nested python dictionaries while keeping the storage intact.

The storage of variables, observables, etc. in env will be organized via nested dictionaries. `NestedDict` class, oftenly used in bundles will be also replaced with `DictWrapper`.

At the current stage we introduce `env.future` storage point, which may be accessed from anywhere and will override observables in the next release.

### Easy nested dictionaries with `DictWrapper`

### Future storage `env.future`

## New packages

### I/O with yaml/pickle/ROOT

### Package `env`

### Package `parameters`

### Package `minimize`

## Other issues

### Deprecate `configuration/`
