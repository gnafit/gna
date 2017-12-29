#ifndef GNACUGPUARRAY_H
#define GNACUGPUARRAY_H

#include "GNAcuGpuMemStates.hh"
#include <iostream>
//#include "../../core/Data.hh"

template <typename T>
class GNAcuGpuArray {
public: 
	GNAcuGpuArray();
        GNAcuGpuArray(T* inArrayPtr, size_t inSize);
	~GNAcuGpuArray();
	void resize(size_t newSize);
	void setByHostArray(T* inHostArr);
        void setByDeviceArray(T* inDeviceArr);
	void setByValue(T value);
	void getContentToCPU(T* dst);
	void getContent(T* dst);
	void transferH2D(); // TODO
	void transferD2H(); // TODO
	T* getArrayPtr() { return devicePtr; }
	size_t getArraySize() { return arrSize; }
	GNAcuGpuArray<T> operator+(GNAcuGpuArray<T> rhs);
        GNAcuGpuArray<T> operator-(GNAcuGpuArray<T> rhs);
        GNAcuGpuArray<T> operator-();
        GNAcuGpuArray<T> operator*(GNAcuGpuArray<T> rhs);
        GNAcuGpuArray<T>& operator=(GNAcuGpuArray<T> rhs);

        GpuMemoryState arrState;
//	DataType type;
protected:
	T* devicePtr;
	T* hostPtr;
        size_t arrSize;
};

#endif /* GNACUGPUARRAY_H */
