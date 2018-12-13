#include "OutputHandle.hh"

using TransformationTypes::OutputHandle;

/**
 * @brief Check the Entry.
 * @copydoc Entry::check()
 */
bool OutputHandle::check() const {
  return m_sink->entry->check() && m_sink->data;
}

/**
 * @brief Dump the Entry.
 * @copydoc Entry::dump()
 */
void OutputHandle::dump() const {
  return m_sink->entry->dump(0);
}

