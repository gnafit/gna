#include "Args.hh"
#include "Sink.hh"
#include "Source.hh"
#include "TransformationErrors.hh"

using TransformationTypes::Source;
using TransformationTypes::Args;
using TransformationTypes::CalculationError;

/**
 * @brief Get i-th Source Data.
 * @param i -- index of a Source.
 * @return i-th Sources's Data as input (const).
 *
 * @exception CalculationError in case invalid index is queried.
 * @exception CalculationError in case input data is not initialized.
 */
const Data<double> &Args::operator[](int i) const {
  if (i < 0 or static_cast<size_t>(i) > m_entry->sources.size()) {
    auto msg = fmt::format("invalid arg idx {0}, have {1} args", i, m_entry->sources.size());
    throw CalculationError(m_entry, msg);
  }
  const Source &src = m_entry->sources[i];
  if (!src.materialized()) {
    auto msg = fmt::format("arg {0} ({1}) have no type on evaluation", i, src.name);
    throw CalculationError(m_entry, msg);
  }
  src.sink->entry->touch();
  return *src.sink->data;
}

