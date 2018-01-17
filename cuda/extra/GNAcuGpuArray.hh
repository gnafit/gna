#ifndef GNACUGPUARRAY_H
#define GNACUGPUARRAY_H

#include "GNAcuDataLocation.hh"
#include <iostream>
//#include "../../core/Data.hh"

template <typename T>
class GNAcuGpuArray {
public: 
	GNAcuGpuArray();
        GNAcuGpuArray(T* inArrayPtr, size_t inSize);
	~GNAcuGpuArray();
	void resize(size_t newSize);
        inline void setSize(size_t inSize) { arrSize = inSize; }
	DataLocation setByHostArray(T* inHostArr);
        DataLocation setByDeviceArray(T* inDeviceArr);
	DataLocation setByValue(T value);
	DataLocation getContentToCPU(T* dst);
	DataLocation getContent(T* dst);
	DataLocation transferH2D(); 
	DataLocation transferD2H(); 
	T* getArrayPtr() { return devicePtr; }
	size_t getArraySize() { return arrSize; }
	GNAcuGpuArray<T> operator+(GNAcuGpuArray<T> &rhs);
        GNAcuGpuArray<T> operator-(GNAcuGpuArray<T> &rhs);
        GNAcuGpuArray<T> operator-();
        GNAcuGpuArray<T> operator*(GNAcuGpuArray<T> &rhs);
        GNAcuGpuArray<T>& operator=(GNAcuGpuArray<T> rhs);

        DataLocation arrState;
//	DataType type;
protected:
	T* devicePtr;
	T* hostPtr;
        size_t arrSize;
};

#endif /* GNACUGPUARRAY_H */
