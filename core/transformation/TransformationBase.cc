#include "TransformationBase.hh"

#include <algorithm>

using TransformationTypes::Entry;
using TransformationTypes::Base;
using TransformationTypes::Accessor;
using TransformationTypes::Initializer;

/**
 * @brief Clone constructor.
 * @copydetails Base::copyEntries
 * @param other -- the other Base.
 */
Base::Base(const Base &other)
  : t_(*this), m_entries(other.m_entries.size())
{
  this->copyEntries(other);
}

/**
 * @brief Clone assignment.
 * @copydetails Base::copyEntries
 * @param other -- the other Base.
 */
Base &Base::operator=(const Base &other) {
  t_ = Accessor(*this);
  m_entries.reserve(other.m_entries.size());
  this->copyEntries(other);
  return *this;
}

/**
 * @brief Clone entries from the other Base.
 *
 * Fills Base::m_entries with clones of Entry instances from the other Base.
 *
 * @param other -- the other Base to copy Entry instances from.
 */
void Base::copyEntries(const Base &other) {
  std::transform(other.m_entries.begin(), other.m_entries.end(),
                 std::back_inserter(m_entries),
                 [this](const Entry &e) { return new Entry{e, this}; });
}

/**
 * @brief Add new Entry.
 * @param e -- new Entry.
 * @return the current number of Entry instances in the Base.
 */
size_t Base::addEntry(Entry *e) {
  size_t idx = m_entries.size();
  m_entries.push_back(e);
  return idx;
}

/**
 * @brief Get an Entry by name.
 * @param name -- Entry name to return.
 * @return Entry.
 * @exception KeyError in case there is no Entry with such a name.
 */
Entry &Base::getEntry(const std::string &name) {
  for (Entry &e: m_entries) {
    if (e.name == name) {
      return e;
    }
  }
  throw KeyError(name, "transformation");
}

/**
 * @brief Initialize the new transformation Entry.
 * @param obj -- the pointer to the TransformationBlock.
 * @param name -- the transformation name.
 * @return transformation Initializer.
 */
template <typename T>
Initializer<T> transformation_(T *obj, const std::string &name) {
  return Initializer<T>(obj, name);
}
