#include "GpuArgs.hh"


void TransformationTypes::GpuArgs::fill_size_vec() {
	for (auto &src : m_cpu_args.m_entry->sources) {
		gpu_sizes_rows.push_back(src.sink->data->gpuArr->rows);
		gpu_sizes_cols.push_back(src.sink->data->gpuArr->columns);
	}
}
