#include "OutputHandle.hh"
#include "TransformationEntry.hh"

#include "config_vars.h"

/**
 * @brief Return pointer to the Sink's data buffer. Evaluate the data if needed in advance.
 * @return pointer to the Sink's data buffer.
 */
template<typename FloatType>
inline const FloatType* TransformationTypes::OutputHandleT<FloatType>::data() const {
  m_sink->entry->touch();
#ifdef GNA_CUDA_SUPPORT
  if (m_sink->data->gpuArr != nullptr) {
     m_sink->data->gpuArr->sync( DataLocation::Host );
  }
#endif
  return this->view();
}

/**
 * @brief Check that Sink depends on a changeable.
 * Simply checks that Entry depends on a changeable.
 * @param x -- changeable to test.
 * @return true if depends.
 */
template<typename FloatType>
inline bool TransformationTypes::OutputHandleT<FloatType>::depends(changeable x) const {
  return m_sink->entry->tainted.depends(x);
}

/**
 * @brief Check the Entry.
 * @copydoc Entry::check()
 */
template<typename FloatType>
bool TransformationTypes::OutputHandleT<FloatType>::check() const {
  return m_sink->entry->check() && m_sink->data;
}

/**
 * @brief Dump the Entry.
 * @copydoc Entry::dump()
 */
template<typename FloatType>
void TransformationTypes::OutputHandleT<FloatType>::dump() const {
  m_sink->entry->dump(0);
}

template class TransformationTypes::OutputHandleT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::OutputHandleT<float>;
#endif

