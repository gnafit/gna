#ifndef GNACUROOTMATH_H
#define GNACUROOTMATH_H

/*#define GnaCuQe 1.602176462e-19
#define GnaCuC  2.99792458e8
#define GnaCuHbar 1.054571596e-34
#define GnaCuKm2MeV(km) (km * 1E-3 * GnaCuQe / (GnaCuHbar * GnaCuC))
*/

#include <cuda.h>
#include <cuda_runtime.h>



template <typename T>
__host__ __device__ __inline__ T Qe() {
	return 1.602176462e-19;
}
// velocity of light
template <typename T>
__host__ __device__ __inline__ T C() {
	return 2.99792458e8;
}  // m s^-1
// Planck's constant
template <typename T>
__host__ __device__ __inline__ T H() {
	return 6.62606876e-34;
}  // J s
// h-bar (h over 2 pi)
template <typename T>
__host__ __device__ __inline__ T Hbar() {
	return 1.054571596e-34;
}  // J s

template <typename T>
__host__ __device__ __inline__ T km2MeV(T km) {
	return km * 1E-3 * Qe<T>() / (Hbar<T>() * C<T>());
}



/*namespace constants {
template <typename T>
__host__ __device__ __inline__ T Qe() { return 1.602176462e-1; }

// velocity of light
template <typename T>
__host__ __device__ __inline__ T C() {  return 2.99792458e8; }  // m s^-1

// Planck's constant
template <typename T>
__host__ __device__ __inline__ T H() {  return 6.62606876e-34;}  // J s

// h-bar (h over 2 pi)
template <typename T>
__host__ __device__ __inline__ T Hbar() {  return 1.054571596e-34;}  // J s

template <typename T>
__host__ __device__ __inline__ T km2MeV(T km) {  return km * 1E-3 * Qe<T>() / (Hbar<T>() * C<T>()); }
}
*/
/*
double Qe();
double C();
double H();
double Hbar();
double km2MeV(double km);
*/

#endif /* GNACUROOTMATH_H */
