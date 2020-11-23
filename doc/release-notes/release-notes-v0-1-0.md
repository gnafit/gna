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

The `DictWrapper` is a wrapper class for dictionaries. Its main function is to handle multi-component case, provided as tuples. For example:
```python
d = {'parent': {'child': 1}}
dw = DictWrapper(d)
print(dw[('parent', 'child')])
```
Will print 1. `DictWrapper` handles getting/setting/checking/etc operations. It also provides nested
iteration.

`DictWrapper` provides a way to automatically split the key based on a character:

```python
d = {'parent': {'child': 1}}
dw = DictWrapper(d, split='.')
print(dw['parent.child'])
```
Will split the key string `'parent.child'` and retrieve a `'child'` from `'parent'` dictionary.

The original dictionary may be retrieved at any time with `dw._obj`.

The dictionary elements are also available via attribute access:
```python
d = {'parent': {'child': 1}}
dw = DictWrapper(d)
print(dw._.parent.child)
```
The only limitation of the attribute access wrapper is that it is unable to create _new_ intermediate
dictionaries.

### Future storage `env.future`

`env.future` is an instance of `DictWrapper` with split key set to '.'. All new UIs are expected to utilize `env.future` for the storage.
At some point `env.future` will replace `env`.

The following UI modules work with `env.future`:
* Dataset organization
    + `dataset-v01-wip`: stores observables in 'spectra' and nuiscance terms in 'pull'.
    + `analysis-v01-wip`: reads 'parameter_groups' for the parameters to build the covariance matrix.
* New UI to facilitate access to `evn`
    + `env-cfg`: controls the `env.future` representation and enables logging.
    + `env-data-root`: copies and converts arrays to ROOT types (TH1/TGraph).
    + `env-data`: provides multiple function to copy `env.future` elements.
    + `env-print`: prints the details of the path in `env.future`.
    + `env-set`: writes arbitrary information into path of `env.future`.
* I/O
    + `save-pickle`: saves subtree of `env.future` to a pickle file.
    + `save-root`: saves ROOT objeects from a subtree of `env.future` to a ROOT file.
    + `save-yaml`: saves subtree of `env.future` to a yaml file.
* Parameters and fitting
    + `pargroup`: selects parameters from `env` and combines them in a group. Stores the group in `env.future['parameter_grous']`.
    + `pargrid`: creates a grid for scanning minimizer. Stores the group in `env.future['pargrid']`.
    + `minimizer-v1`: uses parameter groups from `env.future['parameter_grous']`, stores minimizer in `env.future['minimizer']`.
    + `minimizer-scan`: uses parameter groups from `env.future['parameter_grous']` and
      `env.future['pargrid']`, stores minimizer in `env.future['minimizer']`.
    + `fit-v1`: use minimizer from `env.future['minimizer']` and stores result in
        `env.future['fitresult']` and `env.future['fitresults']`.
* Plotting. All the utilites read inputs from `env.future['spectra']`:
    + `graphviz-v1`
    + `plot-heatmap-v1`
    + `plot-spectrum-v1`

### Common output folders

## New packages and modules

New UI modules and other tools are organized in packages. Some older modules are moved to the
packages as well.

### New packages

* `dataset`: for organizing the analysis.
* `env`: for the operations with `env.future`.
* `legacy`: old unmaintained tools go here.
* `parameters`: tools for working with parameters.
* `minimize`: minimization and fitting tools.
* `plot-v1`: plotting tools.
* `ui`: modules to help with UI.

### New UI modules
* `env`: env and I/O tools
    + `env-cfg`: controls the `env.future` representation and enables logging.
    + `env-cwd`: controls the common output folder. See [Common output folders](#common-output-folders).
    + `env-data`:
    + `env-data-root`: copies and converts arrays to ROOT types (TH1/TGraph).
    + `env-pars-latex`
    + `env-print`
    + `env-set`
    + `save-pickle`
    + `save-root`
    + `save-yaml`
* `parameters`
    + `pargrid`
    + `pargroup`
* `minimize`
    + `minimizer-scan`
    + `minimizer-v1`
* `ui`
    + `cmd-save`
    + `comment`
    + `help`:

New plotting packages:
* `mpl-v1`
* `plot-heatmap-v1`

The following UI modules are a newer version of old ones and utilize `env.future` for the storage:
* `dataset`
    + `dataset-v01-wip`
    + `analysis-v01-wip`
* `minimize`/`fit-v1`
* `plot-v1`
    + `graphviz-v1`
    + `plot-spectrum-v1`

Other packages:
* nmo-set: properly switch NMO under assumption on which mass splitting should be the same.

### I/O with yaml/pickle/ROOT

### Package `env`

### Package `parameters`

### Package `minimize`

## Other issues

### Deprecate `configuration/`
