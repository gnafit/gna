#include "TransformationBase.hh"

#include <algorithm>

using TransformationTypes::EntryT;
using TransformationTypes::BaseT;
using TransformationTypes::Accessor;
using TransformationTypes::Initializer;

/**
 * @brief Clone constructor.
 * @copydetails Base::copyEntries
 * @param other -- the other Base.
 */
template<typename SourceFloatType, typename SinkFloatType>
BaseT<SourceFloatType,SinkFloatType>::BaseT(const BaseT<SourceFloatType,SinkFloatType> &other)
  : t_(*this), m_entries(other.m_entries.size())
{
  this->copyEntries(other);
}

/**
 * @brief Clone assignment.
 * @copydetails Base::copyEntries
 * @param other -- the other Base.
 */
template<typename SourceFloatType, typename SinkFloatType>
BaseT<SourceFloatType,SinkFloatType> &BaseT<SourceFloatType,SinkFloatType>::operator=(const BaseT<SourceFloatType,SinkFloatType> &other) {
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
template<typename SourceFloatType, typename SinkFloatType>
void BaseT<SourceFloatType,SinkFloatType>::copyEntries(const BaseT<SourceFloatType,SinkFloatType> &other) {
  std::transform(other.m_entries.begin(), other.m_entries.end(),
                 std::back_inserter(m_entries),
                 [this](const EntryType &e) { return new EntryType{e, this}; });
}

/**
 * @brief Add new Entry.
 * @param e -- new Entry.
 * @return the current number of Entry instances in the Base.
 */
template<typename SourceFloatType, typename SinkFloatType>
size_t BaseT<SourceFloatType,SinkFloatType>::addEntry(EntryT<SourceFloatType,SinkFloatType> *e) {
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
template<typename SourceFloatType, typename SinkFloatType>
EntryT<SourceFloatType,SinkFloatType> &BaseT<SourceFloatType,SinkFloatType>::getEntry(const std::string &name) {
  for (auto &e: m_entries) {
    if (e.name == name) {
      return e;
    }
  }
  throw KeyError(name, "transformation");
}

template class TransformationTypes::BaseT<double,double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::BaseT<float,float>;
#endif
