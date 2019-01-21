#include "Ints.hh"

using TransformationTypes::Ints;
using TransformationTypes::CalculationError;

/**
 * @brief Get i-th Storage Data.
 * @param i -- index of a Storage.
 * @return i-th Storage's Data.
 *
 * @exception CalculationError in case invalid index is queried.
 * @exception CalculationError in case output data is not initialized.
 */
Data<double> &Ints::operator[](int i) const {
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
CalculationError Ints::error(const std::string &message) const {
  return CalculationError(this->m_entry, message);
}
