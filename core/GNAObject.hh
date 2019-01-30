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

  using SourceFloatType=double;
  using SinkFloatType=double;

  using StorageTypesFunctionArgs = TransformationTypes::StorageTypesFunctionArgsT<SourceFloatType,SinkFloatType>;
  using TypesFunctionArgs = TransformationTypes::TypesFunctionArgsT<SourceFloatType,SinkFloatType>;
  using FunctionArgs = TransformationTypes::FunctionArgsT<SourceFloatType,SinkFloatType>;
  using Args = TransformationTypes::ArgsT<SourceFloatType,SinkFloatType>;
  using Rets = TransformationTypes::RetsT<SourceFloatType,SinkFloatType>;
  using Atypes = TransformationTypes::AtypesT<SourceFloatType,SinkFloatType>;
  using Rtypes = TransformationTypes::RtypesT<SourceFloatType,SinkFloatType>;
  using Function = TransformationTypes::FunctionT<SourceFloatType,SinkFloatType>;
  using TypesFunction = TransformationTypes::TypesFunctionT<SourceFloatType,SinkFloatType>;
  using Entry = TransformationTypes::EntryT<SourceFloatType,SinkFloatType>;
  using Accessor = TransformationTypes::Accessor;
  using HandleT = TransformationTypes::HandleT<SourceFloatType,SinkFloatType>;
  using OutputHandle = TransformationTypes::OutputHandleT<SinkFloatType>;
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

  OutputHandle single() override {
    return (*this)[0].outputs.single();
  }
  bool check() {
    return (*this)[0].check();
  }
  void dump() {
    (*this)[0].dump();
  }
};
