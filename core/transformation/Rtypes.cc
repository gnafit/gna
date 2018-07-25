#include "Rtypes.hh"

#include <stdexcept>

using TransformationTypes::Rtypes;
using TransformationTypes::SinkTypeError;

/**
 * @brief Get i-th Sink DataType.
 *
 * @param i -- Sink index.
 * @return i-th Sink DataType.
 *
 * @exception std::runtime_error in case invalid index is queried.
 */
DataType &Rtypes::operator[](int i) {
  if (i < 0 || static_cast<size_t>(i) >= m_types->size()) {
    throw std::runtime_error(
      (format("invalid access to return type %1%, nsinks: %2%")
              % i % m_types->size()).str());
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
const DataType &Rtypes::operator[](int i) const {
  if (i < 0 || static_cast<size_t>(i) >= m_types->size()) {
    throw std::runtime_error(
      (format("invalid access to return type %1%, nsinks: %2%")
              % i % m_types->size()).str());
  }
  return (*m_types)[i];
}

/**
 * @brief Sink type exception.
 * @param dt -- incorrect DataType.
 * @param message -- exception message.
 * @return exception.
 */
SinkTypeError Rtypes::error(const DataType &dt, const std::string &message) {
  const Sink *sink = nullptr;
  for (size_t i = 0; i < m_types->size(); ++i) {
    if (&(*m_types)[i] == &dt) {
      sink = &m_entry->sinks[i];
      break;
    }
  }
  return SinkTypeError(sink, message);
}

