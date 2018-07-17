#include "Accessor.hh"
#include "EntryHandle.hh"
#include "TransformationBase.hh"
#include "TransformationDebug.hh"

using TransformationTypes::Accessor;
using TransformationTypes::Base;
using TransformationTypes::Handle;

/**
 * @brief Get a Handle for the i-th Entry.
 * @param idx -- index of the Entry.
 * @return Handle for the Entry.
 */
Handle Accessor::operator[](int idx) const {
  return Handle(m_parent->getEntry(idx));
}

/**
 * @brief Get a Handle for the Entry by name.
 * @param name -- Entry's name.
 * @return Handle for the Entry.
 */
Handle Accessor::operator[](const std::string &name) const {
  TR_DPRINTF("accessing %s on %p\n", name.c_str(), (void*)m_parent);
  return Handle(m_parent->getEntry(name));
}

/**
 * @brief Get number of Entry instances.
 * @return number of Entry instances.
 */
size_t Accessor::size() const {
  return m_parent->m_entries.size();
}
