#ifndef GNACUGPUARRAY_H
#define GNACUGPUARRAY_H

#include "GNAcuGpuMemStates.hh"
#include <iostream>

template <typename T>
class GNAcuGpuArray {
public: 
	GpuMemoryState arrState;
	GNAcuGpuArray();
        GNAcuGpuArray(T* inArrayPtr, size_t inSize);
	~GNAcuGpuArray();
	void resize(size_t newSize);
	void setByHostArray(T* inHostArr);
        void setByDeviceArray(T* inDeviceArr);
	void setByValue(T value);
	void getContentToCPU(T* dst);
	void getContent(T* dst);
	T* getArrayPtr() { return arrayPtr; }
	size_t getArraySize() { return arrSize; }
	GNAcuGpuArray<T> operator+(GNAcuGpuArray<T> rhs);
        GNAcuGpuArray<T> operator*(GNAcuGpuArray<T> rhs);
        GNAcuGpuArray<T>& operator=(GNAcuGpuArray<T> rhs);
protected:
	T* arrayPtr;
        size_t arrSize;
};

//template <typename T>
//GNAcuGpuArray<T> operator+(GNAcuGpuArray<T> &lhs, GNAcuGpuArray<T> &rhs) ;
//extern "C" GNAcuGpuArray<double> operator+(GNAcuGpuArray<double> &lhs, GNAcuGpuArray<double> &rhs);


//template <typename T>
//void operator=(GNAcuGpuArray<T>& lhs, const GNAcuGpuArray<T>& rhs);


#endif /* GNACUGPUARRAY_H */
