#include "OutputHandle.hh"
#include "TransformationEntry.hh"

template<typename FloatType>
void TransformationTypes::OutputHandleT<FloatType>::touch() const {
  m_sink->entry->touch_global();
}

/**
 * @brief Return pointer to the Sink's data buffer. Evaluate the data if needed in advance.
 * @return pointer to the Sink's data buffer.
 *
 * The output is expected to be triggered by the user, therefore EntryT::touch_global method
 * is used to trigger TreeManager (if applicable) to update the variables.
 */
template<typename FloatType>
const FloatType* TransformationTypes::OutputHandleT<FloatType>::data() const {
  m_sink->entry->touch_global();

#ifdef GNA_CUDA_SUPPORT
  if(m_sink->entry->getEntryLocation() != DataLocation::Host){
    auto* gpu=m_sink->data->gpuArr.get();
    if (gpu) {
      gpu->sync(DataLocation::Host);
    }
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
bool TransformationTypes::OutputHandleT<FloatType>::depends(changeable x) const {
  return m_sink->entry->tainted.depends(x);
}

/**
 * @brief Check the Entry.
 * @copydoc Entry::check()
 */
template<typename FloatType>
bool TransformationTypes::OutputHandleT<FloatType>::check() const {
  return bool(m_sink->data);
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

