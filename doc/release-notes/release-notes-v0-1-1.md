# Release notes for GNA v0.1.1

This is a bug fix release for the major release of the GNA [v0.1.0](doc/release-notes/release-notes-v0-1-0.md).

In addition to a series of bug fixes, the release includes a few updates. 
The major new feature is the support for the `ccache` for more efficient compilation
in presence of multiple development branches.

The updates include:

[[_TOC_]]

## ccache

GNA now supports [ccache](https://ccache.dev) for caching the object files.
In case `ccache` is installed in the system, it will be used for compilation
which speeds the compilation up significantly when switch between branches.

## New bundles

- `integral_2d1d_v05`: detailed configuration of binning and integration orders
- `reactor_anu_spectra_v06`: provide interpolation segments as outputs
- `rebin_v05`: provide binning as outputs, both histogram and points

## Fixes

- Python3 regressions in the implementation of expressions fixed.
- `analysis-v1` and `dataset-v1` (UIs) now correctly handle the input name.
- `analysis-v1` (UI) now checks that the covariance parameter influences the model.

### Minimization

- Force reinitialization of the parameters.
- Minuit return codes are now decoded and saved as status messages.
- `minimizer-v1` and `minimizer-scan` (UIs) now check that the minimized parameters affect the minimized function.
- Minor changes:
    * Disable static minuit minimizer globally.
    * Legacy minuit minimizer: reset parameters **after** computing minos contours.
