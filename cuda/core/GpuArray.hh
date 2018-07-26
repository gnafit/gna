#ifndef GNACUGPUARRAY_H
#define GNACUGPUARRAY_H

#include "DataLocation.hh"
#include "cuda_config_vars.h"
#include <iostream>
#include "GpuArrayTypes.hh"


/**
 * @brief GPU array, data wrapper.
 *
 * Provides data management system, manage host-device transfers.
 *
 * @author Anna Fatkina
 * @date 2018
 */
template <typename T>
class GpuArray {
public: 
	GpuArray(T* inHostPtr = nullptr);				///< constructor without initialization, sets SyncFlag to Unsynchronized 
        GpuArray(size_t inSize, T* inHostPtr = nullptr); 		///< constructor with initializing, sets SyncFlag to InitializedOnly if there are no errors
	~GpuArray();

        DataLocation Init(size_t inSize, T* inHostPtr = nullptr); 	///< Initialization
        inline void setSize(size_t inSize) { arrSize = inSize; }	
	/**
	* Copies data (H2D) from inHostArr to GPU array pointer.
	*/
	DataLocation setByHostArray(T* inHostArr);
	/**
	* Copies data (D2D) from inDeviceArr to GPU array pointer.
	*/
        DataLocation setByDeviceArray(T* inDeviceArr);
	/**
	* Sets all GPU array elements by value.
	*/
	DataLocation setByValue(T value);
	/**
	* Copies data (D2H) from device to host internal pointer.
	*/
	DataLocation getContentToCPU(T* dst);
	/**
	* Copies data from internal device pointer to dst.
	*/
	DataLocation getContent(T* dst);
	void sync_H2D(); 
	void sync_D2H();
	/**
	* Makes data at location loc relevant.
	* Syncronaze if not synchronized, else do nothing.
	*/
	void sync(DataLocation loc);
	/**
	* Make data relevant both on GPU and CPU
	*/
	void synchronize(); 
	T* getArrayPtr() { return devicePtr; }
        inline void setArrayPtr(T* inDevPtr) {devicePtr = inDevPtr; }
	size_t getArraySize() { arrSize = rows * columns; return arrSize; }

        void negate();
	GpuArray<T>& operator+=(GpuArray<T> &rhs);
        GpuArray<T>& operator-=(GpuArray<T> &rhs);
        GpuArray<T>& operator*=(GpuArray<T> &rhs);
        GpuArray<T>& operator*=(T rhs);
        GpuArray<T> operator=(GpuArray<T> rhs);
	void dump(); 						///< Shows data from GPU

	inline void setLocation( DataLocation loc ) { dataLoc = loc; syncFlag =  SyncFlag::Unsynchronized; }

//        DataLocation arrState;
//	DataType type;
	T* devicePtr;
	T* hostPtr;
        size_t arrSize;
	size_t rows;
	size_t columns;
	StorageOrder order;
	ArrayType type;
	bool deviceMemAllocated{false};
	DataLocation dataLoc;         ///< Shows where actual data is placed or whether it inited or crashed.	
	SyncFlag syncFlag;            ///< May be Synchronized (the same data on CPU and GPU), Unsynchronized (not the same data) or SyncFailed (copied with error)
};

#endif /* GNACUGPUARRAY_H */
