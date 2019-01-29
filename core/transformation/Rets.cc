#include "Rets.hh"

using TransformationTypes::EntryT;
using TransformationTypes::Rets;
using TransformationTypes::CalculationError;

/**
 * @brief Get i-th Sink Data.
 * @param i -- index of a Sink.
 * @return i-th Sink's Data as output.
 *
 * @exception CalculationError in case invalid index is queried.
 * @exception CalculationError in case output data is not initialized.
 */
Data<double> &Rets::operator[](int i) const {
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
CalculationError<EntryT<double,double>> Rets::error(const std::string &message) const {
  return CalculationError<EntryT<double,double>>(this->m_entry, message);
}
