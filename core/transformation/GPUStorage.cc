#include "GPUStorage.hh"

void TransformationTypes::GPUStorage::fill_size_vec() {
	for (auto& src : m_entry->sources) {
		gpu_sizes_rows.push_back(src.sink->data->gpuArr->rows);
		gpu_sizes_cols.push_back(src.sink->data->gpuArr->columns);
	}
}

void TransformationTypes::GPUStorage::initGPUStorage() {
	size_t tmp_size = size();
	double** tmp = (double**)malloc(tmp_size * sizeof(double*));
	for (size_t i = 0; i < tmp_size; i++) {
		tmp[i] =
		    m_entry->sources[i].sink->data->gpuArr->devicePtr;
	}
	copyH2D(m_gpu_args, tmp, tmp_size);
	fill_size_vec();
}
