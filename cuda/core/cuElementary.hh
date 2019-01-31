#pragma once

template<typename T>
void cusum(T** in, T** out, unsigned int N, unsigned int M);

template<typename T>
void cuproduct(T** in, T** out, unsigned int N, unsigned int M, unsigned int** argshapes);

template<typename T>
void identity_gpu(T** in, T** out, unsigned int N, unsigned int M);
