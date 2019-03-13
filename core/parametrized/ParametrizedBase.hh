#pragma once

#include <string>
#include <type_traits>
#include "EvaluableEntry.hh"
#include "VariableHandle.hh"
#include "EvaluableHandle.hh"
#include "variable.hh"
#include "parameter.hh"
#include "evaluable.hh"
#include "callback.hh"
#include "taintflag.hh"

namespace ParametrizedTypes {
  using VariablesContainer = boost::ptr_vector<ParametrizedEntry>;
  using EvaluablesContainer = boost::ptr_vector<EvaluableEntry>;

  class ParametrizedBase {
    friend class ::ParametersGroup;
    template<typename SourceFloatType, typename SinkFloatType>
    friend class ::GNAObjectT;
  public:
    ParametrizedBase(const ParametrizedBase &other);
    ParametrizedBase &operator=(const ParametrizedBase &other);
    virtual ~ParametrizedBase() {}

  protected:
    ParametrizedBase() : m_taintflag("base", true) { m_taintflag.set_pass_through(); }
    void copyEntries(const ParametrizedBase &other);
    void subscribe_(taintflag flag);

    template <typename T>
    VariableHandle<T> variable_(const std::string &name) {
      auto *e = new ParametrizedEntry(parameter<T>(name.c_str()), name, this);
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
    //template <typename T>
    //dependant<T> evaluable_(const std::string &name,
                            //std::function<T()> func,
                            //const std::vector<int> &sources);
    template <typename T>
    dependant<T> evaluable_(const std::string &name,
                            std::function<T()> func,
                            const std::vector<changeable> &sources);
    template <typename T>
    dependant<T> evaluable_(const std::string &name,
                            size_t size, std::function<void(arrayview<T>&)> vfunc,
                            const std::vector<changeable> &sources);

    taintflag m_taintflag;
  private:
    int findEntry(const std::string &name) const;
    ParametrizedEntry &getEntry(const std::string &name);

    VariableHandle<void> getByField(const variable<void> *field);

    boost::ptr_vector<ParametrizedEntry> m_entries;
    std::vector<callback> m_callbacks;
    boost::ptr_vector<EvaluableEntry> m_eventries;
  };

  template <typename T>
  inline dependant<T>
  ParametrizedBase::evaluable_(const std::string &name,
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
  template <typename T>
  inline dependant<T>
  ParametrizedBase::evaluable_(const std::string &name,
                   size_t size, std::function<void(arrayview<T>&)> vfunc,
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
    dependant<T> dep = dependant<T>(vfunc, sources, name.c_str(), size);
    m_eventries.push_back(new EvaluableEntry{name, depentries, dep, this});
    return dep;
  }
}
