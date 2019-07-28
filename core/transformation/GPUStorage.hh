#pragma once

#include "core/GpuBasics.hh"
#include "GpuArrayTypes.hh"

//#include "Data.hh" 
#include "TransformationEntry.hh"

#include <vector>
#include <iostream>

namespace TransformationTypes {

	struct GPUStorage {

		GPUStorage(const Entry* e ) : m_entry(e) {  std::cout << "Costruct GPUStorage" << std::endl;	}  		///< constructor
	//	const Data<double> &operator[](int i) const { return m_cpu_args[i]; } 	///< get corresponding element in m_cpu_args
		size_t size() const;
		void initGPUStorage();							///< allocates memory for m_gpu_args, transfers it to GPU
	private:
		const Entry* m_entry;
		double** m_gpu_args;
		int initedStorageSize = 0;
		std::vector<size_t> gpu_sizes_rows;
		std::vector<size_t> gpu_sizes_cols;
		void fill_size_vec();	// TODO fill vecs according to gpuArr sizes
	};

}
