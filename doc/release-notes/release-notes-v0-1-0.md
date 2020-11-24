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

## Help, modularity and package organization

### Modularity

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

### Deprecation of the 'configuration/' and 'thirdparty/'

The configuration via `configuration/` is deprecated. All the paths are now
controlled via `$GNAPATH` variable.

The usage of the `thirdparty/` subfolder to organize packages is also deprecatd.

### Help

The help on modules is now better accessible:
* GNA now is able to determine if a `--help` key was used and prints the help without executing
    the other modules, which makes the waiting time much smaller.
* A new UI module `help` was introduced in order to print help on arguments and give some usage
    examples.

## Storage update

### Introduction

With this release we start migrating to the new internal storage format. It is intended to be as
simple and consistent as possible: a set of nested dictionaries. In order to make the handling
easier we introduce a special wrapper `DictWrapper`. Its purpose is only to wrap a set of nested
python dictionaries while keeping the storage intact.

The storage of variables, observables, etc. in env will be organized via nested dictionaries.
`NestedDict` class, often used in bundles will be also replaced with `DictWrapper`.

At the current stage we introduce `env.future` storage point, which may be accessed from anywhere
and will override observables in the next release.

### Easy nested dictionaries with DictWrapper

The `DictWrapper` is a wrapper class for dictionaries. Its main function is to handle
multi-component case, provided as tuples. For example:
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

### Future storage env.future

`env.future` is an instance of `DictWrapper` with split key set to `'.'` (dot). All new UIs are
expected to utilize `env.future` for the storage. At some point `env.future` will replace `env`.

The following UI modules work with `env.future`:
* Dataset organization
    + `dataset-v01-wip`: stores observables in 'spectra' and nuisance terms in 'pull'.
    + `analysis-v01-wip`: reads 'parameter_groups' for the parameters to build the covariance matrix.
* New UI to facilitate access to `evn`
    + `env-cfg`: controls the `env.future` representation and enables logging.
    + `env-data`: provides multiple function to recursively copy `env.future` elements.
    + `env-data-root`: copies and converts arrays to ROOT types (TH1/TGraph).
    + `env-print`: prints the details of the path in `env.future`.
    + `env-set`: writes arbitrary information into path of `env.future`.
* I/O
    + `save-pickle`: saves a subtree of `env.future` to a pickle file.
    + `save-root`: saves ROOT objects from a subtree of `env.future` to a ROOT file.
    + `save-yaml`: saves a subtree of `env.future` to a yaml file.
* Parameters and fitting
    + `pargroup`: selects parameters from `env` and combines them in a group. Stores the group in
      `env.future['parameter_grous']`.
    + `pargrid`: creates a grid for a scanning minimizer. Stores the group in `env.future['pargrid']`.
    + `minimizer-v1`: uses parameter groups from `env.future['parameter_grous']`, stores minimizer
       in `env.future['minimizer']`.
    + `minimizer-scan`: uses parameter groups from `env.future['parameter_grous']` and
      `env.future['pargrid']`, stores minimizer in `env.future['minimizer']`.
    + `fit-v1`: use minimizer from `env.future['minimizer']` and stores result in
        `env.future['fitresult']` and `env.future['fitresults']`.
* Plotting. All the utilities read inputs from `env.future['spectra']`:
    + `graphviz-v1`
    + `plot-heatmap-v1`
    + `plot-spectrum-v1`

### Common output folders

It is often needed to output a set of output files with figures, data, etc. GNA now may handle a
common folder:
* Common folder is created if not available.
* The outputs are stored in the path relative to the common folder. There is no need to specify it
    in all the UI commands.

The following UI commands respect the common folder:
* `env`:
    + `env-cwd`: controls the common output folder.
    + `save-pickle`: saves a subtree of `env.future` to a pickle file.
    + `save-root`: saves ROOT objects from a subtree of `env.future` to a ROOT file.
    + `save-yaml`: saves a subtree of `env.future` to a yaml file.
* `plot-v1`, the output figures are stored in a common folder:
    + `graphviz_v1`
    + `mpl_v1`

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
    + `env-data`: provides multiple function to recursively copy `env.future` elements.
    + `env-data-root`: copies and converts arrays to ROOT types (TH1/TGraph).
    + `env-pars-latex`: print parameters to a latex table.
    + `env-print`: prints the details of the path in `env.future`.
    + `env-set`: writes arbitrary information into path of `env.future`.
    + `save-pickle`: saves a subtree of `env.future` to a pickle file.
    + `save-root`: saves ROOT objects from a subtree of `env.future` to a ROOT file.
    + `save-yaml`: saves a subtree of `env.future` to a yaml file.
* `parameters`: tools to work with parameter groups.
    + `pargroup`: selects parameters from `env` and combines them in a group.
    + `pargrid`: creates a grid for a scanning minimizer.
* `minimize`: minimization and fitting.
    + `minimizer-scan`: provides a hybrid minimizer, which does a scan over a set of parameters and
       minimization over another set of parameters.
* `ui`: UI helping tools.
    + `cmd-save`: saves the whole command to a shell file.
    + `comment`: does nothing essentially. Needed just to keep some text as a comment.
    + `help`: prints a help for an UI command and some usage example.
* 'plot-v1':
    + `mpl-v1`: common commands to work with figures separated to a common UI module.
    + `plot-heatmap-v1`: plots a 2d heatmap.

The following UI modules are a newer version of old ones and utilize `env.future` for the storage:
* `dataset`
    + `dataset-v01-wip`
    + `analysis-v01-wip`
* `minimize`
    + `fit-v1`
    + `minimizer-v1`:
* `plot-v1`
    + `graphviz-v1`
    + `plot-spectrum-v1`

Other packages:
* `nmo-set`: properly switch NMO under assumption on which mass splitting should be the same.

## Detailed description of the new and updated modules

All of the UI modules, described below do have annotated arguments and description provided. The
information may be retrieved via one of the following commands:
```sh
./gna -- <modulename> --help
```
```sh
./gna -- help <modulename>
```
```sh
./gna -- help <modulename> <subcommand>
```

The information, provided via the `help` module at least repeats the information givein in the
release notes, or, sometimes extends it.

### UI

We start from the `ui` package, containing some tools to help with usage. They will be useful for
all the following commands.

#### help

Each UI module recognizes a `--help` option, which prints the module description and available
arguments. The `help` UI module prints the module description and some usage examples. The idea is
imilar to the [tldr](https://tldr.sh).

The following command will print the usage example for the `comment` UI.
```sh
./gna -- help comment
```
```sh
./gna -- help cmd-save
```

Some commands may provide specific examples, which may be retrieved by more detailed argument:

**TBD**

#### comment

Commenting UI. All the arguments are ignored and needed only for annotation.

This module may be used to insert comments into the commandline.

The command will print the arguments upon execution and does nothing more.
```sh
./gna \
        -- comment Initialize a gaussian peak with default configuration and 50 bins \
        -- gaussianpeak --name peak_MC --nbins 50
```

#### cmd-save

Saves the command line to a file. The command then may be repeated and should produce the same output.
The main argument is the output file name to save the command.

Save the whole command to the file 'command.sh':
```sh
./gna \
    -- comment Initialize a gaussian peak with default configuration and 50 bins \
    -- gaussianpeak --name peak_MC --nbins 50 \
    -- cmd-save command.sh
```

### env: working with the environment

The following UI modules are dedicated to working to the future implementation of the environment,
currently located in the `env.future`. We will start with three UI modules `env-cfg`, `env-set` and
`env-print`.

#### env-cfg: controls the representation of the env.future and enables logging.

Global environment configuration UI. Enables verbosity for the debuggin purposes.

All assignments and changes of the environment will be prited to stdout.

Enable verbosity:
```sh
./gna \
    -- env-cfg -v \
    -- gaussianpeak --name peak_MC --nbins 50
```

The output may be filtered with `-x` and `-i` keys. Both support multiple arguments.

The `-x` option excludes matching keys:
```sh
./gna \
    -- env-cfg -v -x fcn \
    -- gaussianpeak --name peak_MC --nbins 50
```

The `-i` option includes matching keys exclusively:
```sh
./gna \
    -- env-cfg -v -i spectrum \
    -- gaussianpeak --name peak_MC --nbins 50
```

#### env-print: prints the details of the path in env.future.

Unlike verbose `env-cfg`, `env-print` UI recursively prints a chosen subtree of the env.

The arguments are paths wihin env to be printed. Paths may contains '.' which will be interpreted as a separator.
It recursively prints key, type of the value and the value.

Print the contents of the subtree 'spectra':
```sh
./gna \
    -- gaussianpeak --name peak_MC --nbins 50 \
    -- env-print spectra
```

The widths of the key and value columns may be set via `-k` and `-l` options respectively.

#### env-set: writes arbitrary information into path of env.future.

Assigns any data within env. Needed to provide an extra information to be saved with `save-yaml` and
`save-pickle`.

The module provides three ways to input data:
1. Update env from a dictionary (nested), defined via YAML.
2. Write a string to an address within env.
3. Write parsed YAML to an address within env.

Optional argument '-r' may be used to set root address.

Write two key-value pairs to the 'test':
```sh
./gna \
    -- env-set -r test '{key1: string, key2: 1.0}' \
    -- env-print test
```
The first value, assigned by the key 'key1' is a string 'string', the second value is a float 1.

The '-y' argument may be used to write a key-value pair:
```sh
./gna \
    -- env-set -r test -y sub '{key1: string, key2: 1.0}' \
    -- env-print test
```
The command does the same, but writes the key-value pairs into a nested dictionary under the key 'sub'.

The '-a' argument simply writes a key-value pair, where value is a string:
```sh
./gna \
    -- env-set -r test -a key1 string \
    -- env-print test
```

### I/O with yaml/pickle/ROOT

#### env-data: provides multiple function to recursively copy env.future elements.

#### env-data-root: copies and converts arrays to ROOT types (TH1/TGraph).

#### env-cwd: controls the common output folder. See [Common output folders](#common-output-folders).

#### save-pickle: saves a subtree of env.future to a pickle file.

#### save-root: saves ROOT objects from a subtree of env.future to a ROOT file.

#### save-yaml: saves a subtree of env.future to a yaml file.

### Working with parameters

#### pargroup: selects parameters from env and combines them in a group.

#### pargrid: creates a grid for a scanning minimizer.

#### env-pars-latex: print parameters to a latex table.

### Package minimize

#### minimizer-v1

#### minimizer-scan: provides a hybrid minimizer, which does a scan over a set of parameters and

#### fit-v1

### Plotting updates

#### mpl-v1: common commands to work with figures separated to a common UI module.

#### plot-heatmap-v1: plots a 2d heatmap.

#### graphviz-v1

#### plot-spectrum-v1

