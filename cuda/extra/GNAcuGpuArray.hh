#ifndef GNACUGPUARRAY_H
#define GNACUGPUARRAY_H

#include "GNAcuDataLocation.hh"
#include <iostream>
//#include "../../core/Data.hh"

template <typename T>
class GNAcuGpuArray {
public: 
	GNAcuGpuArray();
        GNAcuGpuArray(size_t inSize);
	~GNAcuGpuArray();
        DataLocation Init(size_t inSize);
	void resize(size_t newSize);
        inline void setSize(size_t inSize) { arrSize = inSize; }
	DataLocation setByHostArray(T* inHostArr);
        DataLocation setByDeviceArray(T* inDeviceArr);
	DataLocation setByValue(T value);
	DataLocation getContentToCPU(T* dst);
	DataLocation getContent(T* dst);
	DataLocation transferH2D(); 
	void transferD2H(); 
	T* getArrayPtr() { return devicePtr; }
        void setArrayPtr(T* inDevPtr) {devicePtr = inDevPtr; }
	size_t getArraySize() { std::cout << "in size getter " << arrSize << std::endl; return arrSize; }

        void negate();
	GNAcuGpuArray<T>& operator+=(GNAcuGpuArray<T> &rhs);
        GNAcuGpuArray<T>& operator-=(GNAcuGpuArray<T> &rhs);
        GNAcuGpuArray<T>& operator*=(GNAcuGpuArray<T> &rhs);
        GNAcuGpuArray<T>& operator*=(T rhs);
        GNAcuGpuArray<T>& operator=(GNAcuGpuArray<T> rhs);
	void dump();
        DataLocation arrState;
//	DataType type;

	T* devicePtr;
	T* hostPtr;
        size_t arrSize;
};

#endif /* GNACUGPUARRAY_H */
