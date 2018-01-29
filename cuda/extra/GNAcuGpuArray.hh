#ifndef GNACUGPUARRAY_H
#define GNACUGPUARRAY_H

#include "GNAcuDataLocation.hh"
#include "cuda_config_vars.h"
#include <iostream>
//#include "../../core/Data.hh"

template <typename T>
class GNAcuGpuArray {
public: 
	GNAcuGpuArray(T* inHostPtr = nullptr);
        GNAcuGpuArray(size_t inSize, T* inHostPtr = nullptr);
	~GNAcuGpuArray();

        DataLocation Init(size_t inSize, T* inHostPtr = nullptr);
        inline void setSize(size_t inSize) { arrSize = inSize; }
// TODO change returnable value to void, make state getter
	DataLocation setByHostArray(T* inHostArr);
        DataLocation setByDeviceArray(T* inDeviceArr);
	DataLocation setByValue(T value);
	DataLocation getContentToCPU(T* dst);
	DataLocation getContent(T* dst);
	void sync_H2D(); 
	void sync_D2H();
	void sync(DataLocation loc);
	void synchronize(); 
	T* getArrayPtr() { return devicePtr; }
        inline void setArrayPtr(T* inDevPtr) {devicePtr = inDevPtr; }
	size_t getArraySize() { return arrSize; }

        void negate();
	GNAcuGpuArray<T>& operator+=(GNAcuGpuArray<T> &rhs);
        GNAcuGpuArray<T>& operator-=(GNAcuGpuArray<T> &rhs);
        GNAcuGpuArray<T>& operator*=(GNAcuGpuArray<T> &rhs);
        GNAcuGpuArray<T>& operator*=(T rhs);
        GNAcuGpuArray<T> operator=(GNAcuGpuArray<T> rhs);
	void dump() ;

	inline void setLocation( DataLocation loc ) { dataLoc = loc; syncFlag =  Unsynchronized; }

//        DataLocation arrState;
//	DataType type;
	T* devicePtr;
	T* hostPtr;
        size_t arrSize;
	bool deviceMemAllocated{false};
	DataLocation dataLoc;         // Shows where actual data is placed or whether it inited or crashed.	
	SyncFlag syncFlag;            // May be Synchronized (the same data on CPU and GPU), Unsynchronized (not the same data) or SyncFailed (copied with error)
};

#endif /* GNACUGPUARRAY_H */
