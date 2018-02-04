# Dirs:

- extra: contains cuda algorithms implementation, source files for local CMakeLists
- core: contains c++ wrappers that uses algorithms implemented in *extra* -- cuda version of *gna/core* files

# Dependencies:

- gcc 5.x or 6.x (cuda 9.0+)
- cuda toolkit 7.5+

# Uses cuda-based libs:

- cuBLAS


# Build options

To make CUDA availible you should
- have CUDA capable GPU in your target machine, 
- have CUDA Toolkit installed,
- build GNA with option `-DCUDA_SUPPORT=1`

To print debug info you should use `-DCUDA_DEBUG_INFO=<value>` cmake option, where *value* may be the following:
- 1 means that only cuda warnings and errors will be printed,
- 2 means that data management warnings are also printed,
- 3 means that data tranfer messagies are printed.


# Comparative table (Osc Prob Full, vec size: 10 000 000) 

https://git.jinr.ru/gna/gna/commit/8ef13a52f577380d023b3428e6a2751851c61c7c

| Processor                      | Average Time Full (us) | Average Time Computing Only (us) | First Call (us) |  Speed Up (full) | Speed Up (Computing) |
|--------------------------------|------------------------|----------------------------------|-----------------|------------------|----------------------|
| CPU (Intel Core i7) sequential |      1349514.7         |                                  |       ---       |                  |                      |
| GPU (NVIDIA GeForce GTX 970M)  |       220687.4         |                                  |      490959     |        6.12      |                      |
| CPU concurrent (coming soon)   |                        |                                  |                 |                  |                      |


# Comparative table (Osc Prob Full, vec size: 1 000 000)

https://git.jinr.ru/gna/gna/commit/8ef13a52f577380d023b3428e6a2751851c61c7c


| Processor                      | Average Time Full (us) | Average Time Computing Only (us) | First Call (us) |  Speed Up (full) | Speed Up (Computing) |
|--------------------------------|------------------------|----------------------------------|-----------------|------------------|----------------------|
| CPU (Intel Core i7) sequential |       136908.3         |             ----                 |       ---       |                  |                      |
| GPU (NVIDIA GeForce GTX 970M)  |        98733.1         |             5175                 |       ???       |        1.39      |          26.46       |
| CPU concurrent (coming soon)   |                        |                                  |                 |                  |                      |



# Comparative table (Osc Prob Full, vec size: 10 000)


| Processor                      | Average Time Full (us) | Average Time Computing Only (us) | First Call (us) |  Speed Up (full) | Speed Up (Computing) |
|--------------------------------|------------------------|----------------------------------|-----------------|------------------|----------------------|
| CPU (Intel Core i7) sequential |        1669.4          |             ----                 |       ---       |                  |                      |
| GPU (NVIDIA GeForce GTX 970M)  |       98895.3          |                 80               |       ???       |       0.017      |         20.9         |
| CPU concurrent (coming soon)   |                        |                                  |                 |                  |                      |

*Speed up columns are for an accelerating in comparison with sequential CPU version*
