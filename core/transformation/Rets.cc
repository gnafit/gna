#include "Rets.hh"

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
    auto fmt = format("invalid ret idx %1%, have %2% rets");
    throw this->error((fmt % i % m_entry->sinks.size()).str());
  }
  auto &data = m_entry->sinks[i].data;
  if (!data) {
    auto fmt = format("ret %1% (%2%) have no type on evaluation");
    throw this->error((fmt % i % m_entry->sinks[i].name).str());
  }

  return *m_entry->sinks[i].data;
}

/**
 * @brief Calculation error exception.
 * @param message -- exception message.
 * @return exception.
 */
CalculationError Rets::error(const std::string &message) const {
  return CalculationError(this->m_entry, message);
}
