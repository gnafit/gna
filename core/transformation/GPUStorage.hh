#pragma once

#include "GpuBasics.hh"
#include "GpuArrayTypes.hh"

#include <vector>

namespace TransformationTypes {

//	class Entry;

	struct GPUStorage {
		GPUStorage(const Entry* e ) : m_entry(e) {	}  		///< constructor
	//	const Data<double> &operator[](int i) const { return m_cpu_args[i]; } 	///< get corresponding element in m_cpu_args
//		size_t size() const { return m_.size(); }
		void initGPUStorage();							///< allocates memory for m_gpu_args, transfers it to GPU
	private:
		const Entry* m_entry;
		double** m_gpu_args;
		std::vector<size_t> gpu_sizes_rows;
		std::vector<size_t> gpu_sizes_cols;
		void fill_size_vec();	// TODO fill vecs according to gpuArr sizes
	};

}
