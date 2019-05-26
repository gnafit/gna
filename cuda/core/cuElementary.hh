#pragma once

template<typename T>
void cusum(T** in, T** out, unsigned int N, unsigned int M);

template<typename T>
void cuweightedsum(T** in, T** out, T** weights, unsigned int N, unsigned int M, unsigned int nvars);

template<typename T>
void cuweightedsumfill(T** in, T** out, T** weights, T k,  unsigned int N, unsigned int M, unsigned int nvars);

template<typename T>
void cuproduct(T** in, T** out, unsigned int N, unsigned int M);

template<typename T>
void identity_gpu(T** in, T** out, unsigned int N, unsigned int M);


void cufilllike(size_t val, double** out, int N);
void cufilllike(size_t val, float** out, int N);

template<typename T>
void dummy_params_ongpu(T** in, T** out, T* pars, unsigned int inSize, unsigned int npars);

void cuexp(float** array, float** ans_array, int n, int m);
void cuexp(double** array, double** ans_array, int n, int m);

void cupoisson(double** array, double** ans_array, int* n, int amount, int maxn);
void cupoissonapprox(double** array, double** ans_array, int* n, int amount, int maxn);

void cuselfpower(double** array, double** ans_array, int n, int m, double scale);
void cuselfpower(float** array, float** ans_array, int n, int m, float scale);
