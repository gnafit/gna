#include "Rtypes.hh"

#include <stdexcept>

using TransformationTypes::RtypesT;
using TransformationTypes::SinkT;
using TransformationTypes::SinkTypeError;

/**
 * @brief Get i-th Sink DataType.
 *
 * @param i -- Sink index.
 * @return i-th Sink DataType.
 *
 * @exception std::runtime_error in case invalid index is queried.
 */
template<typename SourceFloatType, typename SinkFloatType>
DataType &RtypesT<SourceFloatType,SinkFloatType>::operator[](int i) {
  if (i < 0 || static_cast<size_t>(i) >= m_types->size()) {
    throw std::runtime_error(
      (fmt::format("invalid access to return type {0}, nsinks: {1}", i, m_types->size())));
  }
  return (*m_types)[i];
}

/**
 * @brief Get i-th Sink DataType (const).
 *
 * @param i -- Sink index.
 * @return i-th Sink DataType.
 *
 * @exception std::runtime_error in case invalid index is queried.
 */
template<typename SourceFloatType, typename SinkFloatType>
const DataType &RtypesT<SourceFloatType,SinkFloatType>::operator[](int i) const {
  if (i < 0 || static_cast<size_t>(i) >= m_types->size()) {
    throw std::runtime_error(
      (fmt::format("invalid access to return type {0}, nsinks: {1}", i, m_types->size())));
  }
  return (*m_types)[i];
}

/**
 * @brief Sink type exception.
 * @param dt -- incorrect DataType.
 * @param message -- exception message.
 * @return exception.
 */
template<typename SourceFloatType, typename SinkFloatType>
SinkTypeError<SinkT<SourceFloatType>> RtypesT<SourceFloatType,SinkFloatType>::error(const DataType &dt, const std::string &message) {
  const SinkImpl *sink = nullptr;
  for (size_t i = 0; i < m_types->size(); ++i) {
    if (&(*m_types)[i] == &dt) {
      sink = &m_entry->sinks[i];
      break;
    }
  }
  return SinkTypeError<SinkImpl>(sink, message);
}

template class TransformationTypes::RtypesT<double,double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::RtypesT<float,float>;
#endif
