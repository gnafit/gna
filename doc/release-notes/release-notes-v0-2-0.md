# Release notes for GNA v0.2.0

This is a second version of GNA. The major updates include:

[[_TOC_]]

## Terminal
### Colors
### Logging
### Documentation

## Bundles
### IBD arguments

## UI
- pars
- `ui/cmd_seed`: allow to set seed of randomization to repeat results of MC simulation
- `analysis_v1`: fix of `__extract_obs`, search for `.` in observables (`env.future` style), not `/` (`env.ns` style)
- `pylib/experiments/dayabay`: change concat observable name in expressions
- `pylib/experiments/dayabay_p15a`: change concat observable name in expressions; add `final_concat` observable only once to namespace

### save-xlsx 

### Graphviz
- Add `--plot-sum`
- Collapse long lists

### Parameters
- Collapse long lists

### analysis-v1
- Order

### mpl
- Load style files

### plot-spectrum-v1
- Add `--sqrt` option to plot root square of observable. In case of `--diff` or `--ratio`, it applied only to observable

## Plotting
- Unify functions

## Transformations
### Rectangular integration 2d
### IBD arguments
### Chi2Verbose
- Trnsformation allows to print in `std::cerr` statistics divided by anaysis (stat. part/syst. part of statistics)

## Minimizer
### Compute covariance

## Transformations
### Rectangular integration 2d

## Core
### Dependencies
- Properly ignore frozen dependencies
- Add new Minimizable that allows print every step of minimization to `std::cerr`
- Update requirements

## Tools
### CVMFS
- Add tools to override paths of necessary data before start of experiment

## Docs
- Clean tutorial


### Unit tests
- Integration
