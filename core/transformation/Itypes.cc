#include "Itypes.hh"

#include <stdexcept>

using TransformationTypes::ItypesT;
using TransformationTypes::StorageT;
using TransformationTypes::StorageTypeError;

/**
 * @brief Get i-th Storage DataType.
 *
 * @param i -- Storage index.
 * @return i-th Storage DataType.
 *
 * @exception std::runtime_error in case invalid index is queried.
 */
template<typename SourceFloatType, typename SinkFloatType>
DataType &ItypesT<SourceFloatType,SinkFloatType>::operator[](int i) {
  auto newsize=static_cast<size_t>(i)+1;
  if( newsize > m_types->size() ){
    m_types->resize(newsize);
  }
  else if (i < 0) {
    throw std::runtime_error(
      (fmt::format("invalid access to return type {0}, nstorages: {1}", i, m_types->size())
       ));
  }
  return (*m_types)[i];
}

/**
 * @brief Storage type exception.
 * @param dt -- incorrect DataType.
 * @param message -- exception message.
 * @return exception.
 */
template<typename SourceFloatType, typename SinkFloatType>
StorageTypeError<StorageT<SourceFloatType>> ItypesT<SourceFloatType,SinkFloatType>::error(const DataType &dt, const std::string &message) {
  const StorageImpl *storage = nullptr;
  for (size_t i = 0; i < m_types->size(); ++i) {
    if (&(*m_types)[i] == &dt) {
      storage = &m_entry->storages[i];
      break;
    }
  }
  return StorageTypeError<StorageT<SourceFloatType>>(storage, message);
}

template class TransformationTypes::ItypesT<double,double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::ItypesT<float,float>;
#endif
