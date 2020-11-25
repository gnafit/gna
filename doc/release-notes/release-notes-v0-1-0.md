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

The usage of the `thirdparty/` subfolder to organize packages is also deprecated.

### Help

The help on modules is now better accessible:
* GNA now is able to determine if a `--help` key was used and prints the help without executing
    the other modules, which makes the waiting time much smaller.
* A new UI module `help` was introduced in order to print help on arguments and give some usage
    examples.

Note also, that most of the UI commands support verbosity flag '-v'. Sometimes passing multiple v's
as '-vv' or '-vvv' increases the verbosity.

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
    + `dataset-v1`: stores observables in 'spectra' and nuisance terms in 'pull'.
    + `analysis-v1`: reads 'parameter_groups' for the parameters to build the covariance matrix.
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
    + `env-print`: prints the details of the path in `env.future`.
    + `env-set`: writes arbitrary information into path of `env.future`.
    + `save-pickle`: saves a subtree of `env.future` to a pickle file.
    + `save-root`: saves ROOT objects from a subtree of `env.future` to a ROOT file.
    + `save-yaml`: saves a subtree of `env.future` to a yaml file.
* `parameters`: tools to work with parameter groups.
    + `env-pars-latex`: print parameters to a latex table.
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
    + `dataset-v1`
    + `analysis-v1`
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

The information, provided via the `help` module at least repeats the information given in the
release notes, or, sometimes extends it. While the release notes may be short-spoken the help and
examples will be further extended with GNA development.

In the following examples we often use the option '-vv' to increase verbosity, In practical usage it
may be safely omitted.

### Package UI

We start from the `ui` package, containing some tools to help with usage. They will be useful for
all the following commands.

#### Module help

Each UI module recognizes a `--help` option, which prints the module description and available
arguments. The `help` UI module prints the module description and some usage examples. The idea is
similar to the [tldr](https://tldr.sh).

The following command will print the usage example for the `comment` UI.
```sh
./gna -- help comment
```
```sh
./gna -- help cmd-save
```

Some commands may provide specific examples, which may be retrieved by more detailed argument:

**TBD**

#### Module comment

Commenting UI. All the arguments are ignored and needed only for annotation.

This module may be used to insert comments into the commandline.

The command will print the arguments upon execution and does nothing more.
```sh
./gna \
    -- comment Initialize a gaussian peak with default configuration and 50 bins \
    -- gaussianpeak --name peak_MC --nbins 50
```

#### Module cmd-save

Saves the command line to a file. The command then may be repeated and should produce the same output.
The main argument is the output file name to save the command.

Save the whole command to the file 'command.sh':
```sh
./gna \
    -- comment Initialize a gaussian peak with default configuration and 50 bins \
    -- gaussianpeak --name peak_MC --nbins 50 \
    -- cmd-save command.sh
```

### Package env

The following UI modules are dedicated to working to the future implementation of the environment,
currently located in the `env.future`. We will start with three UI modules `env-cfg`, `env-set` and
`env-print`.

#### Module env-cfg

Global environment configuration UI. Enables verbosity for the debugging purposes.

All assignments and changes of the environment will be printed to stdout.

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

#### Module env-print

Unlike verbose `env-cfg`, `env-print` UI recursively prints a chosen subtree of the env.

The arguments are paths within env to be printed. Paths may contains '.' which will be interpreted as a separator.
It recursively prints key, type of the value and the value.

Print the contents of the subtree 'spectra':
```sh
./gna \
    -- gaussianpeak --name peak_MC --nbins 50 \
    -- env-print spectra
```

The widths of the key and value columns may be set via `-k` and `-l` options respectively.

#### Module env-set

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

### Converting the results

The data of the GNA graph is located in the outputs.

#### Module env-data

Recursively saves outputs as dictionaries with numbers and meta.

The module recursively copies all the outputs from the source location to the target location.
The outputs are converted to the dictionaries. Arrays, shapes, bin edges and object type are saved.
The produced data may then be saved with `save-yaml` and `save-pickle` modules.

Write the data from all the outputs from the 'spectra' to 'output':
```sh
./gna \
    -- gaussianpeak --name peak --nbins 50 \
    -- env-data -c spectra.peak output -vv \
    -- env-print -l 40
```

The last command prints the data to stdout. The value width is limited to 40 symbols.

A common root for source and target paths may be set independently via '-s' and '-t' arguments.
There is also a special argument '-g' to combine graphs by reading X and Y arrays from different outputs.

Store a graph read from 'fcn.x' and 'fcn.y' as 'output.fcn_graph':
```sh
./gna \
    -- gaussianpeak --name peak --nbins 50 \
    -- env-data -s spectra.peak -g fcn.x fcn.y output.fcn_graph \
    -- env-print -l 40
```

Extra information may be saved with data. It should be provided as one ore more YAML dictionaries of the
'-c' and '-g' arguments. The dictionaries will be used to update the target paths.

Provide extra information:
```sh
./gna \
    -- gaussianpeak --name peak --nbins 50 \
    -- env-data -c spectra.peak output '{note: extra information}' -vv \
    -- env-print -l 40
```

#### Module env-data-root

Recursively saves the outputs as ROOT objects: TH1D, TH2D, TGraph.

The module recursively copies all the outputs from the source location to the target location.
The outputs are converted to the ROOT objects. The produced data may then be saved with `save-root` module.

The overall idea is similar to the `env-data` module. Only TH1D, TH2D, TGraph are supported.
While histograms are written automatically for writing graphs the user need to use '-g' argument.

Write the data from all the outputs from the 'spectra' to 'output':
```sh
./gna \
    -- gaussianpeak --name peak --nbins 50 \
    -- env-data-root -c spectra.peak output -vv \
    -- env-print -l 40
```

The last command prints the data to stdout. The value width is limited to 40 symbols.

A common root for source and target paths may be set independently via '-s' and '-t' arguments.

Store a graph read from 'fcn.x' and 'fcn.y' as 'output.fcn_graph':
```sh
./gna \
    -- gaussianpeak --name peak --nbins 50 \
    -- env-data-root -s spectra.peak -g fcn.x fcn.y output.fcn_graph \
    -- env-print -l 40
```

### I/O with yaml, pickle and ROOT

The data, written to `env.future` may be saved to output files: human readable YAML, binary pickle
and binary ROOT. The examples will use the commands from [Converting the results](#converting-the-results)
in order to prepare the data.

#### Module save-yaml

Saves a subtree of the env to a readable YAML file.

The module saves the paths provided as arguments to an output YAML file, provided after '-o' option.
If the outputs should be saved, the data should be converted via `env-data` module.
The YAML is human readable and fits to the purposes of saving a small data samples,
such as fit results or small histograms or graphs.

The module is similar to the modules `save-pickle` and `save-root`.

Write the data, collected in the 'output' to the file 'output.yaml'
```sh
./gna \
    -- gaussianpeak --name peak --nbins 5 \
    -- env-data -c spectra.peak.spectrum output '{note: extra information}' -vv \
    -- env-print -l 40 \
    -- save-yaml output -o output.yaml
```
In this example we have reduced the number of bins in order to improve readability of the 'output.yaml'.

#### Module save-pickle

Saves a subtree of the env to a binary pickle file.

The module saves the paths provided as arguments to an output pickle file, provided after '-o' option.
If the outputs should be saved, the data should be converted via `env-data` module.
The pickle is a binary readable and works fast. It should be preferred over `save-yaml` for the large data.

The module is similar to the modules `save-yaml` and `save-root`.

Write the data, collected in the 'output' to the file 'output.pkl'
```sh
./gna \
    -- gaussianpeak --name peak --nbins 50 \
    -- env-data -c spectra output '{note: extra information}' -vv \
    -- env-print -l 40 \
    -- save-pickle output -o output.pkl
```

#### Module save-root

Saves a subtree of the env to a binary ROOT file.

The module saves the paths provided as arguments to an output ROOT file, provided after '-o' option.
The outputs that should be saved should be converted via `env-data-root` module.

The module is similar to the modules `save-yaml` and `save-pickle`.

Write the data, collected in the 'output' to the file 'output.root'
```sh
./gna \
    -- gaussianpeak --name peak --nbins 50 \
    -- env-data-root -c spectra output \
    -- env-data-root -s spectra.peak -g fcn.x fcn.y output.fcn_graph \
    -- env-print -l 40 \
    -- save-root output -o output.root
```

### Package parameters

#### Module env-pars-latex

Recursively prints parameters as a latex table.

The module enables the user to create a latex table for parameters.
It accepts multiple paths with `env` (not `env.future`) and prints a text table to the stdout
and a latex table to the file, provided after an '-o' option.

Print the parameters to the file 'output.tex':
```sh
./gna \
    -- gaussianpeak --name peak \
    -- env-pars-latex peak -o output.tex
```

The module uses python module [tabulate](https://github.com/astanin/python-tabulate) for printing.

#### Module pargroup

Select a group of parameters for the minimization and other purposes.

The module recursively selects parameters based on their status (free, constrained, fixed)
and inclusion/exclusion mask.
The list is stored in `env.future` and may be used by minimizers.
By default the module selects all the not fixed parameters: free and constrained.

Select not fixed parameters from the namespace 'peak' and store as 'minpars':
```sh
./gna \
    -- gaussianpeak --name peak \
    -- ns --name peak --print \
          --set E0             values=2.5  free \
          --set Width          values=0.3  relsigma=0.2 \
          --set Mu             values=1500 relsigma=0.25 \
          --set BackgroundRate values=1100 fixed \
    -- pargroup minpars peak -vv
```

The '-m' option may be used with few arguments describing the parameter mode. The choices include:
free, constrained and fixed.

Select only _fixed_ parameters from the namespace 'peak' and store as 'minpars':
```sh
./gna \
    -- gaussianpeak --name peak \
    -- ns --name peak --print \
          --set E0             values=2.5  free \
          --set Width          values=0.3  relsigma=0.2 \
          --set Mu             values=1500 relsigma=0.25 \ --set BackgroundRate values=1100 fixed \
    -- pargroup minpars peak -m fixed -vv
```

The parameters may be filtered with '-x' and '-i' flags. The option '-x' will exclude parameters,
full names of which contain one of the string passed as arguments. The option '-i' will include
only matching parameters.

See also: `minimizer-v1`, `minimizer-scan`

#### Module pargrid

Specify a grid for a few parameters to be used with scanning minimizer.

The module provides tools for creating grids to scan over parameters.
It supports `range`, `linspace`, `logspace` and `geomspace` which are similar to their analogues from `numpy`.
It also supports a list of values passed from the command line.

Generate a linear grid for the parameter 'E0':
```sh
./gna -- \
    -- gaussianpeak --name peak \
    -- pargrid scangrid --linspace peak.E0 0.5 4.5 10 -vv
```

The possible options include:
| Option        | Arguments                      | NumPy analogue | Includes end point |
|:--------------|:-------------------------------|:---------------|:-------------------|
| `--range`     | `start` `stop` `step`          | arange         | ✘                  |
| `--linspace`  | `start` `stop` `n`             | linspace       | ✔                  |
| `--geomspace` | `start` `stop` `n`             | geomspace      | ✔                  |
| `--logspace`  | `start_power` `stop_power` `n` | logspace       | ✔                  |
| `--list`      | space separated values         | array          | ✔                  |

Provide a list of grid values from a command line:
```sh
./gna -- \
    -- gaussianpeak --name peak \
    -- pargrid scangrid --linspace peak.E0 1 2 8 -vv
```
### Package dataset

The `dataset` package includes the updates of the UI modules `dataset` and `analysis` used for the
initialization of the minimization functions and statistics.

#### Module dataset-v1

Dataset initialization (v1). Configures the dataset for an experiment.

Dataset defines:
- A pair of theory-data:
    * Observable (model) to be used as fitted function
    * Observable (data) to be fitted to
- Statistical uncertainties (Person/Neyman) [theory/observation]
- Or nuisance parameters

The dataset is added to the `env.future['spectra']`.

By default a theory, fixed at the moment of dataset initialization is used for the stat errors (Pearson's case).

Initialize a dataset 'peak' with a pair of Theory/Data:
```sh
./gna \
    -- gaussianpeak --name peak_MC --nbins 50 \
    -- gaussianpeak --name peak_f  --nbins 50 \
    -- ns --name peak_MC --print \
          --set E0             values=2    fixed \
          --set Width          values=0.5  fixed \
          --set Mu             values=2000 fixed \
          --set BackgroundRate values=1000 fixed \
    -- ns --name peak_f --print \
          --set E0             values=2.5  relsigma=0.2 \
          --set Width          values=0.3  relsigma=0.2 \
          --set Mu             values=1500 relsigma=0.25 \
          --set BackgroundRate values=1100 relsigma=0.25 \
    -- dataset-v1 --name peak --theory-data peak_f.spectrum peak_MC.spectrum -v
```

When a dataset is initialized from a nuisance terms it reads only constrained parameters from the namespace.

Initialize a dataset 'nuisance' with a constrained parameters of 'peak_f':
```sh
./gna \
    -- gaussianpeak --name peak_MC --nbins 50 \
    -- gaussianpeak --name peak_f  --nbins 50 \
    -- ns --name peak_MC --print \
          --set E0             values=2    fixed \
          --set Width          values=0.5  fixed \
          --set Mu             values=2000 fixed \
          --set BackgroundRate values=1000 fixed \
    -- ns --name peak_f --print \
          --set E0             values=2.5  relsigma=0.2 \
          --set Width          values=0.3  relsigma=0.2 \
          --set Mu             values=1500 relsigma=0.25 \
          --set BackgroundRate values=1100 relsigma=0.25 \
    -- dataset-v1 --name nuisance --pull peak_f -v
```

#### Module analysis-v1

### Package minimize

The package `minimize` contains modifications of the already existing `minimizer` and `fit` UI
modules.

#### Module minimizer-v1

Initializes a minimizer for a given statistic and set of parameters.

The module creates a minimizer instance which then may be used for a fit with `fit-v1` module or elsewhere.
The minimizer arguments are: `minimizer name` `statistics` `minpars`. Where:
* `minimizer name` is a name of new minimizer.
* `statistics` is the name of a function to minimizer, which should be created beforehand.
* `minpars` is the name of a parameter group, created by `pargroup`.

Create a minimizer and do a fit of a function 'stats' and a group of parameters 'minpars':
```sh
./gna \
    -- gaussianpeak --name peak_MC --nbins 50 \
    -- gaussianpeak --name peak_f  --nbins 50 \
    -- ns --name peak_MC --print \
          --set E0             values=2    fixed \
          --set Width          values=0.5  fixed \
          --set Mu             values=2000 fixed \
          --set BackgroundRate values=1000 fixed \
    -- ns --name peak_f --print \
          --set E0             values=2.5  relsigma=0.2 \
          --set Width          values=0.3  relsigma=0.2 \
          --set Mu             values=1500 relsigma=0.25 \
          --set BackgroundRate values=1100 relsigma=0.25 \
    -- dataset-v1  --name peak --theory-data peak_f.spectrum peak_MC.spectrum \
    -- analysis-v1 --name analysis --datasets peak \
    -- stats stats --chi2 analysis \
    -- pargroup minpars peak_f -vv \
    -- minimizer-v1 min stats minpars -vv \
    -- fit-v1 min \
    -- env-print fitresult.min
```
The `env-print` will print the status of the minimization, performed by the `fit-v1`.

By default `TMinuit2` minimizer is used from ROOT. The minimizer may be changed with '-t' option to
`scipy` or `minuit` (TMinuit).

Create a minimizer and do a fit of a function 'stats' and a group of parameters 'minpars' using a `scipy` minimizer:
```sh
./gna \
    -- gaussianpeak --name peak_MC --nbins 50 \
    -- gaussianpeak --name peak_f  --nbins 50 \
    -- ns --name peak_MC --print \
          --set E0             values=2    fixed \
          --set Width          values=0.5  fixed \
          --set Mu             values=2000 fixed \
          --set BackgroundRate values=1000 fixed \
    -- ns --name peak_f --print \
          --set E0             values=2.5  relsigma=0.2 \
          --set Width          values=0.3  relsigma=0.2 \
          --set Mu             values=1500 relsigma=0.25 \
          --set BackgroundRate values=1100 relsigma=0.25 \
    -- dataset-v1  --name peak --theory-data peak_f.spectrum peak_MC.spectrum \
    -- analysis-v1 --name analysis --datasets peak \
    -- stats stats --chi2 analysis \
    -- pargroup minpars peak_f -vv \
    -- minimizer-v1 min stats minpars -vv \
    -- fit-v1 min \
    -- env-print fitresult.min
```

#### Module minimizer-scan

#### Module fit-v1

### Plotting updates

#### Module env-cwd

#### Module mpl-v1

#### Module plot-heatmap-v1

#### Module graphviz-v1

#### Module plot-spectrum-v1

