#ifndef PARAMETRIZED_H
#define PARAMETRIZED_H

#include <vector>
#include <string>
#include <functional>

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/noncopyable.hpp>

#include "TObject.h"

#include "Parameters.hh"
#include "SimpleDict.hh"

class VariableDescriptor;
class EvaluableDescriptor;
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

    Entry(const std::string &name, const Base *parent);
    Entry(const Entry &other, const Base *parent);

    void bind(variable<double> var);
    parameter<double> claim(Base *other);

    std::string name;
    bool required;
    double defvalue;
    parameter<double> par;
    variable<double> var;
    State state;
    void *claimer;
    std::vector<variable<double>*> fields;
    const Base *parent;
  };

  class VariableHandle {
    friend class Base;
  public:
    VariableHandle(Entry &entry)
      : m_entry(&entry) { }
    VariableHandle(const VariableHandle &other)
      : VariableHandle(*other.m_entry) { }

    VariableHandle &defvalue(double defvalue);
    const std::string &name() const { return m_entry->name; }
    void bind(variable<double> var) { m_entry->bind(var); }
    parameter<double> claim() { return m_entry->claim(nullptr); }

    void dump() const;
  protected:
    Entry *m_entry;
  };

  typedef boost::view_clone_allocator view_clone_allocator;
  typedef boost::ptr_vector<Entry, view_clone_allocator> SourcesContainer;
  class EvaluableEntry: public boost::noncopyable {
  public:
    EvaluableEntry(const std::string &name, const SourcesContainer &sources,
                   dependant<double> dependant, const Base *parent);
    EvaluableEntry(const EvaluableEntry &other, const Base *parent);

    std::string name;
    SourcesContainer sources;
    dependant<double> dep;
    const Base *parent;
  };

  class EvaluableHandle {
  public:
    EvaluableHandle(EvaluableEntry &entry)
      : m_entry(&entry) { }
    EvaluableHandle(const EvaluableHandle &other)
      : EvaluableHandle(*other.m_entry) { }

    const std::string &name() const { return m_entry-> name; }

    void dump() const;
  protected:
    EvaluableEntry *m_entry;
  };

  typedef boost::ptr_vector<Entry> VariablesContainer;
  typedef boost::ptr_vector<EvaluableEntry> EvaluablesContainer;
  class VariableAccessor {
    friend class Base;
  public:
    VariableAccessor(Base &parent): m_parent(&parent) { }
    variable<double> operator[](const int idx);
    variable<double> operator[](const std::string &name);
    size_t size() const;
  protected:
    VariableAccessor(const VariableAccessor &other);
    VariableAccessor &operator=(const VariableAccessor &other);
  private:
    Base *m_parent;
  };

  class Base {
    friend class VariableAccessor;
    friend class ::ParametersGroup;
    friend class ::GNAObject;
  public:
    Base(const Base &other);
    Base &operator=(const Base &other);

  protected:
    Base() : v_(*this) { }
    void copyEntries(const Base &other);
    void subscribe_(taintflag flag);

    VariableHandle variable_(const std::string &name);
    VariableHandle variable_(variable<double> *field, const std::string &name);

    template <typename T>
    callback callback_(T func) {
      m_callbacks.emplace_back(func, std::vector<changeable>{m_taintflag});
      return m_callbacks.back();
    }
    EvaluableHandle evaluable_(const std::string &name,
                               std::function<double()> func,
                               const std::vector<int> &sources);
    EvaluableHandle evaluable_(const std::string &name,
                               std::function<double()> func,
                               const std::vector<changeable> &sources);

    VariableAccessor v_;

    taintflag m_taintflag;
  private:
    VariableHandle addEntry(const std::string &name);
    int findEntry(const std::string &name) const;
    Entry &getEntry(const std::string &name);

    VariableHandle getByField(const changeable *field);

    boost::ptr_vector<Entry> m_entries;
    std::vector<callback> m_callbacks;
    boost::ptr_vector<EvaluableEntry> m_eventries;
  };

  inline variable<double> VariableAccessor::operator[](const int idx) {
    return m_parent->m_entries[idx].var;
  }

  inline variable<double> VariableAccessor::operator[](const std::string &nm) {
    return *m_parent->getEntry(nm).fields[0];
  }

  inline size_t VariableAccessor::size() const {
    return m_parent->m_entries.size();
  }
}

class VariableDescriptor: public TObject,
                          public ParametrizedTypes::VariableHandle {
public:
  typedef ParametrizedTypes::VariableHandle BaseClass;

  VariableDescriptor(const BaseClass &other)
    : VariableHandle(other), name(BaseClass::name())
  { }
  VariableDescriptor(const VariableDescriptor &other)
    : VariableDescriptor(BaseClass(other))
  { }
  VariableDescriptor(ParametrizedTypes::Entry &entry)
    : VariableDescriptor(BaseClass(entry))
  { }
  static VariableDescriptor invalid(const std::string name);

  void bind(variable<double> var) { BaseClass::bind(var); }
  parameter<double> claim() { return BaseClass::claim(); }
  bool isFree() const;
  bool required() const { return m_entry->required; }
  double defvalue() const { return m_entry->defvalue; }

  std::string name;

  ClassDef(VariableDescriptor, 0);
};

class EvaluableDescriptor: public TObject,
                           public ParametrizedTypes::EvaluableHandle {
public:
  typedef ParametrizedTypes::EvaluableHandle BaseClass;

  typedef ParametrizedTypes::SourcesContainer SourcesContainer;
  typedef SimpleDict<VariableDescriptor, SourcesContainer> Sources;

  EvaluableDescriptor(const BaseClass &other)
    : EvaluableHandle(other), name(BaseClass::name()),
      sources(m_entry->sources)
  { }
  EvaluableDescriptor(const EvaluableDescriptor &other)
    : EvaluableDescriptor(BaseClass(other))
  { }
  EvaluableDescriptor(ParametrizedTypes::EvaluableEntry &entry)
    : EvaluableDescriptor(BaseClass(entry))
  { }
  static EvaluableDescriptor invalid(const std::string name);

  variable<double> &get();

  const std::string name;
  const Sources sources;

  ClassDef(EvaluableDescriptor, 0);
};

#endif // PARAMETRIZED_H
