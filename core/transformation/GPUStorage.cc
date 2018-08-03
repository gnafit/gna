#include "GPUStorage.hh"
#include "core/GpuBasics.hh"


void TransformationTypes::GPUStorage::fill_size_vec() {
	for (auto& src : m_entry->sources) {
		gpu_sizes_rows.push_back(src.sink->data->gpuArr->rows);
		gpu_sizes_cols.push_back(src.sink->data->gpuArr->columns);
	}
}

void TransformationTypes::GPUStorage::initGPUStorage() {
	if (inited) return;
	int  tmp_size = static_cast<int>(size());
	double** tmp = (double**)malloc(tmp_size * sizeof(double*));
	for (int i = 0; i < tmp_size; i++) {
		std::cerr << "i = " << i << std::endl;
		tmp[i] =
		    m_entry->sources[i].sink->data->gpuArr->devicePtr;
	}
	copyH2D(m_gpu_args, tmp, tmp_size);
	fill_size_vec();
	inited = true;
}

size_t TransformationTypes::GPUStorage::size() const { 
	return m_entry->sources.size(); 
}
