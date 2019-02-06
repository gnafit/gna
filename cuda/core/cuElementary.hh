#pragma once

template<typename T>
void cusum(T** in, T** out, unsigned int N, unsigned int M);

template<typename T>
void cuweightedsum(T** in, T** out, T* weights, unsigned int N, unsigned int M);

template<typename T>
void cuproduct(T** in, T** out, unsigned int N, unsigned int M);

template<typename T>
void identity_gpu(T** in, T** out, unsigned int N, unsigned int M);

template<typename T>
void dummy_params_ongpu(T** in, T** out, T* pars, unsigned int inSize, unsigned int npars);
