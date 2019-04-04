#pragma once

#include <string>
#include <stdexcept>

#include "VariableHandle.hh"
#include "VariableDescriptor.hh"
#include "parameter.hh"
#include "variable.hh"

//namespace GNA{
    class VariableDescriptor: public ParametrizedTypes::VariableHandle<void> {
    public:
      using BaseClass = ParametrizedTypes::VariableHandle<void>;

      VariableDescriptor(const BaseClass &other)
        : BaseClass(other)
      { }
      VariableDescriptor(const VariableDescriptor &other) = default;
      VariableDescriptor(ParametrizedTypes::ParametrizedEntry &entry)
        : BaseClass(entry)
      { }
      static VariableDescriptor invalid(const int& idx) {
          throw std::runtime_error(
          (fmt::format("Variable: invalid entry, idx == `{0}'", idx)));

      };
      static VariableDescriptor invalid(const std::string name) {
        throw std::runtime_error(
          (fmt::format("Variable: invalid entry, name == `{0}'", name)));
      }

      void bind(variable<void> var) { BaseClass::bind(var); }
      parameter<void> claim() { return BaseClass::claim(); }
      bool isFree() const {
        return BaseClass::m_entry->state == ParametrizedTypes::ParametrizedEntry::State::Free;
      }
      bool required() const { return BaseClass::m_entry->required; }
      std::string typeName() const {
        return BaseClass::m_entry->var.typeName();
      }
    };
//}
