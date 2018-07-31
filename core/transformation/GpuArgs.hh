#pragma once

#include "Args.hh"
#include "Atypes.hh"
#include "GpuBasics.hh"
#include "GpuArrayTypes.hh"

#include <vector>

namespace TransformationTypes {

	class Entry;

	struct GpuArgs {
		GpuArgs(const Args &cpu_args) : m_cpu_args(cpu_args) {	// TODO make ptr to array of ptrs
			size_t tmp_size = size();
			double** tmp = (double**)malloc( tmp_size *sizeof(double*) );
			size_t a_size = size();
			for (size_t i = 0; i < a_size; i++) {
				tmp[i] = cpu_args.m_entry->sources[i].sink->data->gpuArr->devicePtr;
			}
			copyH2D(m_gpu_args, tmp, tmp_size);
			fill_size_vec();
		}
		const Data<double> &operator[](int i) const { return m_cpu_args[i]; } //TODO get corresponding element in m_cpu_args
		size_t size() const { return m_cpu_args.size(); }
	private:
		const Args m_cpu_args;
		double** m_gpu_args;
		std::vector<size_t> gpu_sizes_rows;
		std::vector<size_t> gpu_sizes_cols;
		void fill_size_vec();	// TODO fill vecs according to gpuArr sizes
	};

}
