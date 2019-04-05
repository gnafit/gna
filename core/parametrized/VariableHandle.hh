#pragma once

#include <iostream>
#include "fmt/format.h"
#include "ParametrizedEntry.hh"

namespace ParametrizedTypes {
  template <typename T>
  class VariableHandle;

  template <>
  class VariableHandle<void> {
    friend class ParametrizedBase;
  public:
    VariableHandle(ParametrizedEntry &entry)
      : m_entry(&entry) { }
    VariableHandle(const VariableHandle<void> &other)
      : VariableHandle(*other.m_entry) { }
    const std::string &name() const { return m_entry->name; }
    VariableHandle<void> required(bool req) {
      m_entry->required = req;
      return *this;
    }
    size_t hash() const {return m_entry->par.hash();}
    void dump() const;

	variable<void> getVariable() { return m_entry->var; }
  protected:
    void bind(variable<void> var) { m_entry->bind(var); }
    parameter<void> claim() { return m_entry->claim(nullptr); }

    ParametrizedEntry *m_entry;
  };

  inline void VariableHandle<void>::dump() const {
    std::cerr << m_entry->name << ", ";
    switch (m_entry->state) {
    case ParametrizedEntry::Free:
      std::cerr << "free";
      break;
    case ParametrizedEntry::Bound:
      std::cerr << "bound";
      break;
    case ParametrizedEntry::Claimed:
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
    friend class ParametrizedBase;
    using BaseClass = VariableHandle<void>;
  public:
    VariableHandle(ParametrizedEntry &entry)
      : BaseClass(entry) { }
    VariableHandle(const VariableHandle<T> &other)
      : BaseClass(other) { }
    explicit VariableHandle(const VariableHandle<void> &other)
      : BaseClass(other) { }
    VariableHandle<T> required(bool req) {
      m_entry->required = req;
      return *this;
    }
    void bind(variable<T> var) { m_entry->bind(var); }
    parameter<T> claim() {
      return parameter<T>(m_entry->claim(nullptr));
    }

	variable<T> getVariable() { return variable<T>(m_entry->var); }
  };
}
