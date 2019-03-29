#include "ParametrizedBase.hh"

using ParametrizedTypes::ParametrizedEntry;
using ParametrizedTypes::VariableHandle;
using ParametrizedTypes::ParametrizedBase;

ParametrizedBase::ParametrizedBase(const ParametrizedBase &other)
  : m_taintflag(other.m_taintflag),
    m_callbacks(other.m_callbacks)
{
  copyEntries(other);
}

ParametrizedBase &ParametrizedBase::operator=(const ParametrizedBase &other) {
  copyEntries(other);
  m_taintflag = other.m_taintflag;
  m_callbacks = other.m_callbacks;
  return *this;
}

void ParametrizedBase::copyEntries(const ParametrizedBase &other) {
  std::transform(other.m_entries.begin(), other.m_entries.end(),
                 std::back_inserter(m_entries),
                 [this](const ParametrizedEntry &e) {
                   return new ParametrizedEntry{e, this};
                 });
  std::transform(other.m_eventries.begin(), other.m_eventries.end(),
                 std::back_inserter(m_eventries),
                 [this](const EvaluableEntry &e) {
                   return new EvaluableEntry{e, this};
                 });

}

void ParametrizedBase::subscribe_(taintflag flag) {
  m_taintflag.subscribe(flag);
}

int ParametrizedBase::findEntry(const std::string &name) const {
  for (size_t i = 0; i < m_entries.size(); ++i) {
    if (name == m_entries[i].name) {
      return i;
    }
  }
  return -1;
}

ParametrizedEntry &ParametrizedBase::getEntry(const std::string &name) {
  int i = findEntry(name);
  if (i < 0) {
    throw std::runtime_error(
      fmt::format("unknown parameter `{0}'", name)
      );
  }
  return m_entries[i];
}

VariableHandle<void> ParametrizedBase::getByField(const variable<void> *field) {
  for (auto &e: m_entries) {
    if (e.field == field) {
      return VariableHandle<void>(e);
    }
  }
  throw std::runtime_error("unable to find variable entry by field");
}
