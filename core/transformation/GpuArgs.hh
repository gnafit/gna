#pragma once

#include "GpuBasics.hh"
#include "GpuArrayTypes.hh"

#include <vector>

namespace TransformationTypes {

	class Entry;

	struct GpuArgs {
		GpuArgs(const Entry *e) : m_entry(e) {	}  		///< constructor
		const Data<double> &operator[](int i) const { return m_entry->sources[i]; } 	///< get corresponding element in m_cpu_args
		size_t size() const { return entry->gpustorage.size(); }
	private:
		const Entry *m_entry; 
	};

}
