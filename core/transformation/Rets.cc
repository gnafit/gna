#include "Rets.hh"

#include "TransformationEntry.hh"

using TransformationTypes::EntryT;
using TransformationTypes::RetsT;
using TransformationTypes::CalculationError;

/**
 * @brief Get i-th Sink Data.
 * @param i -- index of a Sink.
 * @return i-th Sink's Data as output.
 *
 * @exception CalculationError in case invalid index is queried.
 * @exception CalculationError in case output data is not initialized.
 */
template<typename SourceFloatType, typename SinkFloatType>
Data<SinkFloatType> &RetsT<SourceFloatType,SinkFloatType>::operator[](int i) const {
  if (i < 0 or static_cast<size_t>(i) > m_entry->sinks.size()) {
    auto msg = fmt::format("invalid ret idx {0}, have {1} rets", i, m_entry->sinks.size());
    throw this->error(msg);
  }
  auto &data = m_entry->sinks[i].data;
  if (!data) {
    auto msg = format("ret {0} ({1}) have no type on evaluation", i, m_entry->sinks[i].name);
    throw this->error(msg);
  }

  return *m_entry->sinks[i].data;
}

/**
 * @brief Calculation error exception.
 * @param message -- exception message.
 * @return exception.
 */
template<typename SourceFloatType, typename SinkFloatType>
CalculationError<EntryT<SourceFloatType,SinkFloatType>> RetsT<SourceFloatType,SinkFloatType>::error(const std::string &message) const {
  return CalculationError<EntryT<SourceFloatType,SinkFloatType>>(this->m_entry, message);
}

/**
 * @brief Get number of transformation sinks.
 * @return Number of transformation Sink instances.
 */
template<typename SourceFloatType, typename SinkFloatType>
size_t RetsT<SourceFloatType,SinkFloatType>::size() const {
  return m_entry->sinks.size();
}

/**
 * @brief Freeze the Entry.
 *
 * While entry is frozen the taintflag is not propagated. Entry is always up to date.
 */
template<typename SourceFloatType, typename SinkFloatType>
void RetsT<SourceFloatType,SinkFloatType>::freeze()  {
  m_entry->tainted.freeze();
}

/**
 * @brief Untaint the Entry.
 *
 * Set Entry's taintflag to false.
 */
template<typename SourceFloatType, typename SinkFloatType>
void RetsT<SourceFloatType,SinkFloatType>::untaint()  {
  m_entry->tainted=false;
}

/**
 * @brief Unfreeze the Entry.
 *
 * Enables the taintflag propagation.
 */
template<typename SourceFloatType, typename SinkFloatType>
void RetsT<SourceFloatType,SinkFloatType>::unfreeze()  {
  m_entry->tainted.unfreeze();
}

template class TransformationTypes::RetsT<double,double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::RetsT<float,float>;
#endif
