#include <stdexcept>
#include <iostream>

#include <boost/format.hpp>
using boost::format;

#include "Parametrized.hh"

using ParametrizedTypes::Entry;
using ParametrizedTypes::VariableHandle;
using ParametrizedTypes::EvaluableEntry;
using ParametrizedTypes::EvaluableHandle;
using ParametrizedTypes::Base;

Entry::Entry(parameter<void> par, const std::string &name, const Base *parent)
  : name(name), required(true),
    par(par), var(par), state(Free), claimer(nullptr),
    parent(parent)
{
}

Entry::Entry(const Entry &other, const Base *parent)
  : Entry(other.par, other.name, parent)
{
}

void Entry::bind(variable<void> newvar) {
  if (!var.is(par)) {
    throw std::runtime_error(
      (format("can not rebind parameter `%1%'") % name).str()
      );
  }
  var.replace(newvar);
  if (field != &var) {
    field->assign(newvar);
  }
  state = Entry::State::Bound;
}

parameter<void> Entry::claim(Base *other) {
  if (state != Entry::State::Free) {
    throw std::runtime_error(
      (format("claiming non-free parameter `%1%'") % name).str()
      );
  }
  state = Entry::State::Claimed;
  claimer = other;
  return par;
}

EvaluableEntry::EvaluableEntry(const std::string &name,
                               const SourcesContainer &sources,
                               dependant<void> dependant,
                               const Base *parent)
  : name(name), sources(sources), dep(dependant), parent(parent)
{
}

EvaluableEntry::EvaluableEntry(const EvaluableEntry &other,
                               const Base *parent)
  : EvaluableEntry(other.name, other.sources, other.dep, parent)
{
}

Base::Base(const Base &other)
  : m_taintflag(other.m_taintflag),
    m_callbacks(other.m_callbacks)
{
  copyEntries(other);
}

Base &Base::operator=(const Base &other) {
  copyEntries(other);
  m_taintflag = other.m_taintflag;
  m_callbacks = other.m_callbacks;
  return *this;
}

void Base::copyEntries(const Base &other) {
  std::transform(other.m_entries.begin(), other.m_entries.end(),
                 std::back_inserter(m_entries),
                 [this](const Entry &e) {
                   return new Entry{e, this};
                 });
  std::transform(other.m_eventries.begin(), other.m_eventries.end(),
                 std::back_inserter(m_eventries),
                 [this](const EvaluableEntry &e) {
                   return new EvaluableEntry{e, this};
                 });

}

void Base::subscribe_(taintflag flag) {
  m_taintflag.subscribe(flag);
}

int Base::findEntry(const std::string &name) const {
  for (size_t i = 0; i < m_entries.size(); ++i) {
    if (name == m_entries[i].name) {
      return i;
    }
  }
  return -1;
}

Entry &Base::getEntry(const std::string &name) {
  int i = findEntry(name);
  if (i < 0) {
    throw std::runtime_error(
      (format("unknown parameter `%1%'") % name).str()
      );
  }
  return m_entries[i];
}

VariableHandle<void> Base::getByField(const variable<void> *field) {
  for (auto &e: m_entries) {
    if (e.field == field) {
      return VariableHandle<void>(e);
    }
  }
  throw std::runtime_error("unable to find variable entry by field");
}
