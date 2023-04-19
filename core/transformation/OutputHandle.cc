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

/**
 * @brief Fill the data from buffer.
 *
 * Fills n elements of the data array. If n>n_max, fills n_max.
 * Notifies the subscribers, but keeps the entry not tainted.
 *
 * @param n -- number of elements to write.
 * @param data -- input data.
 * @return -- number of elements written.
 */
template<typename FloatType>
size_t TransformationTypes::OutputHandleT<FloatType>::fill(size_t n, FloatType* data) const {
  if(!m_sink->materialized()){
    return 0;
  }
  auto* entry=m_sink->entry;

  // Ensure entry is up-to-date and not tainted
  entry->touch_global();

  // Save the frozen state and freeze as the entry should not propagate the taintflag manually
  // Will raise exception if entry is tainted
  entry->tainted.freeze();
  entry->tainted.taint();

  auto& target=*m_sink->data;
  size_t n_writable=std::min(n, target.type.size());
  if(n_writable){
    target.x.head(n_writable) = typename Data<FloatType>::ArrayViewType(data, n_writable);
  }

  // Notify descendants
  m_sink->entry->tainted.notify();

  return n_writable;
}

template class TransformationTypes::OutputHandleT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::OutputHandleT<float>;
#endif

