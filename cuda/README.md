# Dirs:

- extra: contains cuda algorithms implementation, source files for local CMakeLists
- core: contains c++ wrappers that uses algorithms implemented in *extra* -- cuda version of *gna/core* files

# Dependencies:

- gcc 5.x
- cuda toolkit 7.5+

# Uses cuda-based libs:

- cuBLAS


# Comparative table (Osc Prob Full, vec size: 10 000 000) #8ef13a52f577380d023b3428e6a2751851c61c7c

| Processor                      | Average Time Full (us) | Average Time Computing Only (us) | First Call (us) |  Speed Up (full) | Speed Up (Computing) |
|--------------------------------|------------------------|----------------------------------|-----------------|------------------|----------------------|
| CPU (Intel Core i7) sequential |      1349514.7         |                                  |       ---       |                  |                      |
| GPU (NVIDIA GeForce GTX 970M)  |       220687.4         |                                  |      490959     |        6.12      |                      |
| CPU concurrent (coming soon)   |                        |                                  |                 |                  |                      |


# Comparative table (Osc Prob Full, vec size: 1 000 000)

| Processor                      | Average Time Full (us) | Average Time Computing Only (us) | First Call (us) |  Speed Up (full) | Speed Up (Computing) |
|--------------------------------|------------------------|----------------------------------|-----------------|------------------|----------------------|
| CPU (Intel Core i7) sequential |       136908.3         |                                  |       ---       |                  |                      |
| GPU (NVIDIA GeForce GTX 970M)  |        98733.1         |                                  |       ???       |        1.39      |                      |
| CPU concurrent (coming soon)   |                        |                                  |                 |                  |                      |



*Speed up columns are for an accelerating in comparison with sequential CPU version*
