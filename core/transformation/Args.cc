#include "Args.hh"
#include "Sink.hh"
#include "Source.hh"
#include "TransformationErrors.hh"

using TransformationTypes::Source;
using TransformationTypes::Args;
using TransformationTypes::CalculationError;

/**
 * @brief Get i-th Source Data. If CUDA enabled and relevant data is placed on GPU, it synchronizes data before return it. 
 * @param i -- index of a Source.
 * @return i-th Sources's Data as input (const).
 *
 * @exception CalculationError in case invalid index is queried.
 * @exception CalculationError in case input data is not initialized.
 */
const Data<double> &Args::operator[](int i) const {
  if (i < 0 or static_cast<size_t>(i) > m_entry->sources.size()) {
    auto fmt = format("invalid arg idx %1%, have %2% args");
    throw CalculationError(m_entry, (fmt % i % m_entry->sources.size()).str());
  }
  const Source &src = m_entry->sources[i];
  if (!src.materialized()) {
    auto fmt = format("arg %1% (%2%) have no type on evaluation");
    throw CalculationError(m_entry, (fmt % i % src.name).str());
  }
  src.sink->entry->touch();
#ifdef GNA_CUDA_SUPPORT
  std::cout << "before sync" << std::endl;
  if (src.sink->data->gpuArr) {
	std::cout << "in if" << std::endl;
    src.sink->data->gpuArr->sync(this->m_entry->getEntryLocation());
  }
#endif
  return *src.sink->data;
}

