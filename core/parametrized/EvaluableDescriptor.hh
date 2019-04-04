#pragma once

#include <string>
#include <stdexcept>
#include "fmt/format.h"

#include "SimpleDict.hh"
#include "EvaluableEntry.hh"
#include "EvaluableHandle.hh"
#include "VariableDescriptor.hh"
#include "variable.hh"

//namespace GNA {
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
          (fmt::format("Evaluable: invalid entry, name == `{0}'", name)));
      }

      variable<void> &get() { return BaseClass::m_entry->dep; }
      std::string typeName() const {
        return BaseClass::m_entry->dep.typeName();
      }

      const Sources sources;
    };
//}

