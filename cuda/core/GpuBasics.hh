#pragma once

//extern "C" 

template<typename T>
void copyH2D(T* dst, T* src, int N);

template<typename T>
void copyH2D_NOALL(T* dst, T* src, int N);

template<typename T>
void cuwr_free(T* ptr); 
