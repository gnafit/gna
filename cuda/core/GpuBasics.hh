#pragma once
// extern "C" { 

template<typename T>
void copyH2D(T* dst, T* src, unsigned int N);

template<typename T>
void copyH2D_NOALL(T* dst, T* src, unsigned int N) ;

template<typename T>
void cuwr_free(T* ptr);

//}

/*
template<> void copyH2D<double>(void* dst, void* src, unsigned int N);
template<> void copyH2D_NOALL<double>(void* dst, void* src, unsigned int N) ;
template<> void cuwr_free<double>(void* ptr);

template<> void copyH2D<unsigned int>(void* dst, void* src, unsigned int N);
template<> void copyH2D_NOALL<unsigned int>(void* dst, void* src, unsigned int N) ;
template<> void cuwr_free<unsigned int>(void* ptr);
*/




//extern template void copyH2D<double>(double*, double*, unsigned int);
//extern template void copyH2D<unsigned int>(unsigned int*, unsigned int*, unsigned int);
//template void copyH2D(double* dst, double* src, int N);

//extern "C" void copyH2Dui(unsigned int* dst, unsigned int* src, unsigned int N);
//extern "C" void copyH2Dd(double* dst, double* src, unsigned int N);

/*extern "C" void copyH2Dui(unsigned int* dst, unsigned int* src, unsigned int N) {
	copyH2D<unsigned int>(dst, src, N);
}

extern "C" void copyH2Dd(double* dst, double* src, unsigned int N) {
	copyH2D<double>(dst, src, N);
}*/
