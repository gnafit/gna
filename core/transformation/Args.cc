#include "Args.hh"
#include "Sink.hh"
#include "Source.hh"
#include "TransformationErrors.hh"

#include "config_vars.h"

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
    auto msg = fmt::format("invalid arg idx {0}, have {1} args", i, m_entry->sources.size());
    throw CalculationError(m_entry, msg);
  }
  const Source &src = m_entry->sources[i];
  if (!src.materialized()) {
    auto msg = fmt::format("arg {0} ({1}) have no type on evaluation", i, src.name);
    throw CalculationError(m_entry, msg);
  }
  src.sink->entry->touch();
#ifdef GNA_CUDA_SUPPORT
  if (src.sink->data->gpuArr) {
    src.sink->data->gpuArr->sync(this->m_entry->getEntryLocation());
  }
#endif
  return *src.sink->data;
}

/**
 * @brief Touch all the sources
 */
void Args::touch() const {

  for(auto& source: m_entry->sources){
    if (!source.materialized()) {
      auto msg = fmt::format("arg ({1}) have no type on evaluation", source.name);
      throw CalculationError(m_entry, msg);
    }
    source.sink->entry->touch();
#ifdef GNA_CUDA_SUPPORT
    auto& gpuarr=source.sink->data->gpuArr;
    if (gpuarr) {
      gpuarr->sync(this->m_entry->getEntryLocation());
    }
#endif
  }
}

