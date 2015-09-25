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

using Sources = EvaluableDescriptor::Sources;

Entry::Entry(const std::string &name, const Base *parent)
  : name(name), required(true),
    par(name.c_str()), var(par), state(Free), claimer(nullptr),
    parent(parent)
{
}

Entry::Entry(const Entry &other, const Base *parent)
  : Entry(other.name, parent)
{
}

void Entry::bind(variable<double> newvar) {
  if (!var.is(par)) {
    throw std::runtime_error(
      (format("can not rebind parameter `%1%'") % name).str()
      );
  }
  var.replace(newvar);
  for (variable<double> *field: fields) {
    field->assign(newvar);
  }
  state = Entry::State::Bound;
}

parameter<double> Entry::claim(Base *other) {
  if (state != Entry::State::Free) {
    throw std::runtime_error(
      (format("claiming non-free parameter `%1%'") % name).str()
      );
  }
  state = Entry::State::Claimed;
  claimer = other;
  return par;
}

VariableHandle &VariableHandle::defvalue(double defvalue) {
  if (m_entry->state != Entry::State::Free) {
    throw std::runtime_error(
      (format("setting default value for non-free paremeter `%1%'")
       % m_entry->name).str());
  }
  m_entry->defvalue = defvalue;
  m_entry->par = defvalue;
  m_entry->required = false;
  return *this;
}

void VariableHandle::dump() const {
  std::cerr << m_entry->name << ", ";
  switch (m_entry->state) {
  case Entry::Free:
    std::cerr << "free";
    break;
  case Entry::Bound:
    std::cerr << "bound";
    break;
  case Entry::Claimed:
    std::cerr << "claimed";
    break;
  default:
    std::cerr << "UNKNOWN STATE";
    break;
  }
  std::cerr << std::endl;
}

EvaluableEntry::EvaluableEntry(const std::string &name,
                               const SourcesContainer &sources,
                               dependant<double> dependant,
                               const Base *parent)
  : name(name), sources(sources), dep(dependant), parent(parent)
{
}

EvaluableEntry::EvaluableEntry(const EvaluableEntry &other,
                               const Base *parent)
  : EvaluableEntry(other.name, other.sources, other.dep, parent)
{
}

void EvaluableHandle::dump() const {
  std::cerr << m_entry->name;
  std::cerr << std::endl;
}

bool VariableDescriptor::isFree() const {
  return m_entry->state == Entry::State::Free;
}

variable<double> &EvaluableDescriptor::get() {
  return m_entry->dep;
}

Base::Base(const Base &other)
  : v_(*this), m_taintflag(other.m_taintflag),
    m_callbacks(other.m_callbacks)
{
  copyEntries(other);
}

Base &Base::operator=(const Base &other) {
  copyEntries(other);
  v_ = VariableAccessor(*this);
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

VariableHandle Base::addEntry(const std::string &name) {
  Entry *e = new Entry(name, this);
  m_entries.push_back(e);
  e->par.subscribe(m_taintflag);
  return VariableHandle(*e);
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

VariableHandle Base::getByField(const changeable *field) {
  for (auto &e: m_entries) {
    if (e.fields.empty()) {
      continue;
    }
    if (e.fields[0] == field) {
      return VariableHandle(e);
    }
  }
  throw std::runtime_error("unable to find variable entry by field");
}

VariableHandle Base::variable_(const std::string &name) {
  return addEntry(name);
}

VariableHandle Base::variable_(variable<double> *field,
                               const std::string &name) {
  auto handle = variable_(name);
  field->assign(handle.m_entry->par);
  handle.m_entry->fields.push_back(field);
  return handle;
}

EvaluableHandle Base::evaluable_(const std::string &name,
                                 std::function<double()> func,
                                 const std::vector<int> &sources) {
  std::vector<changeable> deps;
  SourcesContainer depentries;
  for (int varid: sources) {
    Entry &e = m_entries[varid];
    deps.push_back(e.var);
    depentries.push_back(&e);
  }
  dependant<double> dep = dependant<double>(func, deps, name.c_str());
  m_eventries.push_back(new EvaluableEntry{name, depentries, dep, this});
  return EvaluableHandle(m_eventries.back());
}

EvaluableHandle Base::evaluable_(const std::string &name,
                                 std::function<double()> func,
                                 const std::vector<changeable> &sources) {
  std::vector<int> srcs;
  for (changeable chdep: sources) {
    size_t i;
    for (i = 0; i < m_entries.size(); ++i) {
      if (m_entries[i].var.is(chdep)) {
        break;
      }
    }
    if (i < m_entries.size()) {
      srcs.push_back(i);
    } else {
      throw std::runtime_error(
        (format("evaluable `%1%': unknown dependancy `%1%'")
         % name % chdep.name()).str()
        );
    }
  }
  return evaluable_(name, func, srcs);
}

VariableDescriptor VariableDescriptor::invalid(const std::string name) {
  throw std::runtime_error(
    (format("Variable: invalid entry, name == `%1%'") % name).str());
}

EvaluableDescriptor EvaluableDescriptor::invalid(const std::string name) {
  throw std::runtime_error(
    (format("Evaluable: invalid entry, name == `%1%'") % name).str());
}
