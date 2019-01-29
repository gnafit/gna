#pragma once

#include "Parametrized.hh"
#include "TransformationDescriptor.hh"
#include "TransformationBase.hh"
#include "TransformationBind.hh"
#include "GPUFunctionArgs.hh"

class GNAObject: public virtual TransformationTypes::Base,
                 public virtual ParametrizedTypes::Base {
public:
  using VariablesContainer = ParametrizedTypes::VariablesContainer;
  using EvaluablesContainer = ParametrizedTypes::EvaluablesContainer;
  using TransformationsContainer = TransformationTypes::EntryContainer;
  using Variables = SimpleDict<VariableDescriptor, VariablesContainer>;
  using Evaluables = SimpleDict<EvaluableDescriptor, EvaluablesContainer>;
  using Transformations = SimpleDict<TransformationDescriptor, TransformationsContainer>;

  GNAObject()
    : TransformationTypes::Base(),
      ParametrizedTypes::Base(),
      variables(ParametrizedTypes::Base::m_entries),
      evaluables(m_eventries),
      transformations(TransformationTypes::Base::m_entries)
    { }
  GNAObject(const GNAObject &other)
    : TransformationTypes::Base(other),
      ParametrizedTypes::Base(other),
      variables(ParametrizedTypes::Base::m_entries),
      evaluables(m_eventries),
      transformations(TransformationTypes::Base::m_entries)
    { }

  void subscribe(taintflag flag) {
    ParametrizedTypes::Base::subscribe_(flag);
  }

  TransformationDescriptor operator[](size_t idx) {
    return TransformationDescriptor(TransformationTypes::Base::getEntry(idx));
  }

  TransformationDescriptor operator[](const std::string &name) {
    return TransformationDescriptor(TransformationTypes::Base::getEntry(name));
  }

  void dumpObj();

  Variables variables;
  Evaluables evaluables;
  Transformations transformations;
protected:
  class SingleTransformation { };
  GNAObject(SingleTransformation)
    : TransformationTypes::Base(1),
      ParametrizedTypes::Base(),
      variables(ParametrizedTypes::Base::m_entries),
      evaluables(m_eventries),
      transformations(TransformationTypes::Base::m_entries)
  { }

  using StorageTypesFunctionArgs = TransformationTypes::StorageTypesFunctionArgs;
  using TypesFunctionArgs = TransformationTypes::TypesFunctionArgs;
  using FunctionArgs = TransformationTypes::FunctionArgs;
  using Args = TransformationTypes::Args;
  using Rets = TransformationTypes::Rets;
  using Atypes = TransformationTypes::Atypes;
  using Rtypes = TransformationTypes::Rtypes;
  using Function = TransformationTypes::Function;
  using TypesFunction = TransformationTypes::TypesFunction;
  using Entry = TransformationTypes::Entry;
  using Accessor = TransformationTypes::Accessor;
  using Handle = TransformationTypes::Handle;
};

class GNASingleObject: public GNAObject,
                       public SingleOutput {
public:
  GNASingleObject()
    : GNAObject(SingleTransformation())
  { }
  GNASingleObject(const GNASingleObject &other)
    : GNAObject(other)
  { }

  TransformationTypes::OutputHandle single() override {
    return (*this)[0].outputs.single();
  }
  bool check() {
    return (*this)[0].check();
  }
  void dump() {
    (*this)[0].dump();
  }
};
