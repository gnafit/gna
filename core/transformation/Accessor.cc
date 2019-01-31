#include "Accessor.hh"
#include "EntryHandle.hh"
#include "TransformationBase.hh"
#include "TransformationDebug.hh"

using TransformationTypes::AccessorT;
using TransformationTypes::HandleT;

/**
 * @brief Get a Handle for the i-th Entry.
 * @param idx -- index of the Entry.
 * @return Handle for the Entry.
 */
template<typename SourceFloatType, typename SinkFloatType>
HandleT<SourceFloatType,SinkFloatType> AccessorT<SourceFloatType,SinkFloatType>::operator[](int idx) const {
  return HandleT<SourceFloatType,SinkFloatType>(m_parent->getEntry(idx));
}

/**
 * @brief Get a Handle for the Entry by name.
 * @param name -- Entry's name.
 * @return Handle for the Entry.
 */
template<typename SourceFloatType, typename SinkFloatType>
HandleT<SourceFloatType,SinkFloatType> AccessorT<SourceFloatType,SinkFloatType>::operator[](const std::string &name) const {
  TR_DPRINTF("accessing %s on %p\n", name.c_str(), (void*)m_parent);
  return HandleT<SourceFloatType,SinkFloatType>(m_parent->getEntry(name));
}

/**
 * @brief Get number of Entry instances.
 * @return number of Entry instances.
 */
template<typename SourceFloatType, typename SinkFloatType>
size_t AccessorT<SourceFloatType,SinkFloatType>::size() const {
  return m_parent->m_entries.size();
}

template class TransformationTypes::AccessorT<double,double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::AccessorT<float,float>;
#endif
