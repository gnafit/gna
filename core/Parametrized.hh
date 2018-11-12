#pragma once

#include <vector>
#include <string>
#include <functional>
#include <iostream>

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/noncopyable.hpp>

#include <boost/format.hpp>

#include "SimpleDict.hh"
#include "dependant.hh"
#include "taintflag.hh"
#include "callback.hh"

class ParametersGroup;
class GNAObject;
namespace ParametrizedTypes {
  class Base;
  class Entry: public boost::noncopyable {
    friend class Base;
  public:
    enum State {
      Free = 0, Claimed, Bound,
    };

    Entry(parameter<void> par, std::string name, const Base *parent);
    Entry(const Entry &other, const Base *parent);

    void bind(variable<void> var);
    parameter<void> claim(Base *other);

    std::string name;
    bool required;
    parameter<void> par;
    variable<void> var;
    State state;
    void *claimer;
    variable<void> *field;
    const Base *parent;
  };

  template <typename T>
  class VariableHandle;

  template <>
  class VariableHandle<void> {
    friend class Base;
  public:
    VariableHandle(Entry &entry)
      : m_entry(&entry) { }
    VariableHandle(const VariableHandle<void> &other)
      : VariableHandle(*other.m_entry) { }
    const std::string &name() const { return m_entry->name; }
    VariableHandle<void> required(bool req) {
      m_entry->required = req;
      return *this;
    }
    void dump() const;
  protected:
    void bind(variable<void> var) { m_entry->bind(var); }
    parameter<void> claim() { return m_entry->claim(nullptr); }

    Entry *m_entry;
  };

  inline void VariableHandle<void>::dump() const {
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

  template <typename T>
  class VariableHandle: public VariableHandle<void> {
    friend class Base;
    using BaseClass = VariableHandle<void>;
  public:
    VariableHandle(Entry &entry)
      : BaseClass(entry) { }
    VariableHandle(const VariableHandle<T> &other)
      : BaseClass(other) { }
    explicit VariableHandle(const VariableHandle<void> &other)
      : BaseClass(other) { }
    VariableHandle<T> required(bool req) {
      m_entry->required = req;
      return *this;
    }
    VariableHandle<T> &defvalue(const T& defvalue);
    void bind(variable<T> var) { m_entry->bind(var); }
    parameter<T> claim() {
      return parameter<T>(m_entry->claim(nullptr));
    }
  };

  template <typename T>
  VariableHandle<T> &VariableHandle<T>::defvalue(const T& defvalue) {
    if (m_entry->state != Entry::State::Free) {
      throw std::runtime_error(
        (boost::format("setting default value for non-free paremeter `%1%'")
         % m_entry->name).str());
    }
    m_entry->par = defvalue;
    m_entry->required = false;
    return *this;
  }

  using view_clone_allocator = boost::view_clone_allocator;
  using SourcesContainer = boost::ptr_vector<Entry, view_clone_allocator>;
  class EvaluableEntry: public boost::noncopyable {
  public:
    EvaluableEntry(std::string name, const SourcesContainer &sources,
                   dependant<void> dependant, const Base *parent);
    EvaluableEntry(const EvaluableEntry &other, const Base *parent);

    std::string name;
    SourcesContainer sources;
    dependant<void> dep;
    const Base *parent;
  };

  template <typename T>
  class EvaluableHandle {
  public:
    EvaluableHandle<T>(EvaluableEntry &entry)
      : m_entry(&entry) { }
    EvaluableHandle<T>(const EvaluableHandle<T> &other)
      : EvaluableHandle<T>(*other.m_entry) { }

    const std::string &name() const { return m_entry->name; }

    void dump() const;
  protected:
    EvaluableEntry *m_entry;
  };

  template <typename T>
  inline void EvaluableHandle<T>::dump() const {
    std::cerr << m_entry->name;
    std::cerr << std::endl;
  }

  using VariablesContainer = boost::ptr_vector<Entry>;
  using EvaluablesContainer = boost::ptr_vector<EvaluableEntry>;

  class Base {
    friend class ::ParametersGroup;
    friend class ::GNAObject;
  public:
    Base(const Base &other);
    Base &operator=(const Base &other);

  protected:
    Base() = default;
    void copyEntries(const Base &other);
    void subscribe_(taintflag flag);

    template <typename T>
    VariableHandle<T> variable_(const std::string &name) {
      auto *e = new Entry(parameter<T>(name.c_str()), name, this);
      m_entries.push_back(e);
      e->par.subscribe(m_taintflag);
      e->field = &e->var;
      return VariableHandle<T>(*e);
    }
    template <typename T>
    VariableHandle<T> variable_(variable<T> *field, const std::string &name) {
      auto handle = variable_<T>(name);
      field->assign(handle.m_entry->par);
      handle.m_entry->field = field;
      return handle;
    }

    template <typename T>
    callback callback_(T func) {
      m_callbacks.emplace_back(func, std::vector<changeable>{m_taintflag});
      return m_callbacks.back();
    }
    template <typename T>
    dependant<T> evaluable_(const std::string &name,
                            std::function<T()> func,
                            const std::vector<int> &sources);
    template <typename T>
    dependant<T> evaluable_(const std::string &name,
                            std::function<T()> func,
                            const std::vector<changeable> &sources);

    taintflag m_taintflag;
  private:
    int findEntry(const std::string &name) const;
    Entry &getEntry(const std::string &name);

    VariableHandle<void> getByField(const variable<void> *field);

    boost::ptr_vector<Entry> m_entries;
    std::vector<callback> m_callbacks;
    boost::ptr_vector<EvaluableEntry> m_eventries;
  };

  template <typename T>
  inline dependant<T>
  Base::evaluable_(const std::string &name,
                   std::function<T()> func,
                   const std::vector<changeable> &sources)
  {
    SourcesContainer depentries;
    for (changeable chdep: sources) {
      size_t i;
      for (i = 0; i < m_entries.size(); ++i) {
        if (m_entries[i].var.is(chdep)) {
          depentries.push_back(&m_entries[i]);
        }
      }
    }
    DPRINTFS("make evaluable: %i deps", int(sources.size()));
    dependant<T> dep = dependant<T>(func, sources, name.c_str());
    m_eventries.push_back(new EvaluableEntry{name, depentries, dep, this});
    return dep;
  }
}

class VariableDescriptor: public ParametrizedTypes::VariableHandle<void> {
public:
  using BaseClass = ParametrizedTypes::VariableHandle<void>;

  VariableDescriptor(const BaseClass &other)
    : BaseClass(other)
  { }
  VariableDescriptor(const VariableDescriptor &other) = default;
  VariableDescriptor(ParametrizedTypes::Entry &entry)
    : BaseClass(entry)
  { }
  static VariableDescriptor invalid(const int& idx) {
      throw std::runtime_error(
      (boost::format("Variable: invalid entry, idx == `%1%'") % idx).str());

  };
  static VariableDescriptor invalid(const std::string name) {
    throw std::runtime_error(
      (boost::format("Variable: invalid entry, name == `%1%'") % name).str());
  }

  void bind(variable<void> var) { BaseClass::bind(var); }
  parameter<void> claim() { return BaseClass::claim(); }
  bool isFree() const {
    return BaseClass::m_entry->state == ParametrizedTypes::Entry::State::Free;
  }
  bool required() const { return BaseClass::m_entry->required; }
  std::string typeName() const {
    return BaseClass::m_entry->var.typeName();
  }

};

class EvaluableDescriptor: public ParametrizedTypes::EvaluableHandle<void> {
public:
  using BaseClass = ParametrizedTypes::EvaluableHandle<void>;

  using SourcesContainer = ParametrizedTypes::SourcesContainer;
  using Sources = SimpleDict<VariableDescriptor, SourcesContainer>;

  EvaluableDescriptor(const BaseClass &other)
    : BaseClass(other),
      sources(BaseClass::m_entry->sources)
  { }
  EvaluableDescriptor(const EvaluableDescriptor &other)
    : EvaluableDescriptor(BaseClass(other))
  { }
  EvaluableDescriptor(ParametrizedTypes::EvaluableEntry &entry)
    : EvaluableDescriptor(BaseClass(entry))
  { }
  static EvaluableDescriptor invalid(const std::string name) {
    throw std::runtime_error(
      (boost::format("Evaluable: invalid entry, name == `%1%'") % name).str());
  }

  variable<void> &get() { return BaseClass::m_entry->dep; }
  std::string typeName() const {
    return BaseClass::m_entry->dep.typeName();
  }

  const Sources sources;
};
