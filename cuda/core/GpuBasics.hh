#pragma once

template<typename T>
void copyH2D_ALL(T* &dst, T* src, unsigned int N);

template<typename T>
void copyH2D_NA(T* dst, T* src, unsigned int N) ;

template<typename T>
void copyH2D_NA(T** dst, T* src, size_t N) ;

template<typename T>
void cuwr_free(T* &ptr);

template<typename T>
void device_malloc(T* &dst, unsigned int N);

template<typename T> 
void debug_drop(T* dst, unsigned int N);

template<typename T> 
void debug_drop(T** in, size_t M /*how many arrs*/ , size_t N /*length of single arr*/ );
