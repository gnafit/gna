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
    auto fmt = format("invalid ret idx %1%, have %2% ints");
    throw this->error((fmt % i % m_entry->storages.size()).str());
  }
  auto &data = m_entry->storages[i].data;
  if (!data) {
    auto fmt = format("ret %1% (%2%) have no type on evaluation");
    throw this->error((fmt % i % m_entry->storages[i].name).str());
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
