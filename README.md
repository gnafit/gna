# Global Neutrino Analysis Project

[![Docs status](https://git.jinr.ru/gna/gna/badges/master/pipeline.svg)](https://git.jinr.ru/gna/gna/commits/master)

<!-- Current version is 0.1.1, see the [v0.1.0 release notes](doc/release-notes/release-notes-v0-1-0.md) -->
<!-- and the [bugfix release notes](doc/release-notes/release-notes-v0-1-1.md). -->

Global Neutrino Analysis (GNA) project aims at providing the following tools for
the needs of the data analysis related to the neutrino physics:

*  Build complex physical models with large number of parameters using dataflow
   principles.
*  Implicit caching and lazy evaluation.
*  High performance fitting using CPU/CPU (multithreading)/GPU.
   Switching between modes at a runtime, not at a compile time.
*  Statistical analysis of data of neutrino experiments.
*  Combined analysis.

The project is on the alpha stage.

# Links

| header                    | header                                     |
| ------                    | ------                                     |
| Homepage                  | https://astronu.jinr.ru/wiki/index.php/GNA |
| Documentation             | http://gna.pages.jinr.ru/gna/              |
| Public mirror (master)    | https://github.com/gnafit/gna              |
| Main repository (private) | https://git.jinr.ru/gna/gna                |

# Input data

Most of the experiment related input data is restricted and located in a separate
repository.

External data repositories should be available under `data/` subfolder of the
GNA root.

| Experiment and dataset  | Repository                                 | Comments |
| ------                  | ------                                     | --- |
| Daya Bay                | https://git.jinr.ru/gna/data_dayabay       | |
| JUNO                    | https://git.jinr.ru/gna/data_juno          | |
| Common                  | https://git.jinr.ru/gna-public/data-common | public |
