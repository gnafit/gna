#include "Ints.hh"
#include "TransformationEntry.hh"

using TransformationTypes::EntryT;
using TransformationTypes::IntsT;
using TransformationTypes::CalculationError;

/**
 * @brief Get i-th Storage Data.
 * @param i -- index of a Storage.
 * @return i-th Storage's Data.
 *
 * @exception CalculationError in case invalid index is queried.
 * @exception CalculationError in case output data is not initialized.
 */
template<typename SourceFloatType, typename SinkFloatType>
Data<SourceFloatType> &IntsT<SourceFloatType,SinkFloatType>::operator[](int i) const {
  if (i < 0 or static_cast<size_t>(i) > m_entry->storages.size()) {
    auto msg = fmt::format("invalid ret idx {0}, have {1} ints", i, m_entry->storages.size());
    throw this->error(msg);
  }
  auto &data = m_entry->storages[i].data;
  if (!data) {
    auto msg = format("ret {0} ({1}) have no type on evaluation", i, m_entry->storages[i].name);
    throw this->error(msg);
  }

  return *m_entry->storages[i].data;
}

/**
 * @brief Calculation error exception.
 * @param message -- exception message.
 * @return exception.
 */
template<typename SourceFloatType, typename SinkFloatType>
CalculationError<EntryT<SourceFloatType,SinkFloatType>> IntsT<SourceFloatType,SinkFloatType>::error(const std::string &message) const {
  return CalculationError<EntryType>(this->m_entry, message);
}

/**
 * @brief Get number of transformation storages.
 * @return Number of transformation storage instances.
 */
template<typename SourceFloatType, typename SinkFloatType>
size_t IntsT<SourceFloatType,SinkFloatType>::size() const {
  return m_entry->storages.size();
}

template class TransformationTypes::IntsT<double,double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::IntsT<float,float>;
#endif
