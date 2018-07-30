#include "GpuArgs.hh"


void GpuArgs::fill_size_vec() {
	for (auto &arg : cpu_args) {
		gpu_sizes_rows.push_back(tmp[i] = arg->m_entry->gpuArr->rows);
		gpu_sizes_cols.push_back(tmp[i] = arg->m_entry->gpuArr->columns);
	}
}
