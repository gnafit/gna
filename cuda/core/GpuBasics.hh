#pragma once

template<typename T>
void copyH2D(T* &dst, T* src, unsigned int N);

template<typename T>
void copyD2D_NA(T* dst, T* src, unsigned int N) ;

template<typename T>
void cuwr_free(T* &ptr);

