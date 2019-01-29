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

  using StorageTypesFunctionArgs = TransformationTypes::StorageTypesFunctionArgsT<double,double>;
  using TypesFunctionArgs = TransformationTypes::TypesFunctionArgsT<double,double>;
  using FunctionArgs = TransformationTypes::FunctionArgsT<double,double>;
  using Args = TransformationTypes::ArgsT<double,double>;
  using Rets = TransformationTypes::RetsT<double,double>;
  using Atypes = TransformationTypes::AtypesT<double,double>;
  using Rtypes = TransformationTypes::RtypesT<double,double>;
  using Function = TransformationTypes::FunctionT<double,double>;
  using TypesFunction = TransformationTypes::TypesFunctionT<double,double>;
  using Entry = TransformationTypes::EntryT<double,double>;
  using Accessor = TransformationTypes::Accessor;
  using Handle = TransformationTypes::Handle;
  using OutputHandle = TransformationTypes::OutputHandleT<double>;
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
