#pragma once

#include "Parametrized.hh"
#include "TransformationDescriptor.hh"
#include "TransformationBase.hh"
#include "TransformationBind.hh"

class GNAObject: public ParametrizedTypes::Base,
                 public TransformationTypes::Base {
public:
  typedef ParametrizedTypes::VariablesContainer VariablesContainer;
  typedef ParametrizedTypes::EvaluablesContainer EvaluablesContainer;
  typedef TransformationTypes::EntryContainer TransformationsContainer;
  typedef SimpleDict<VariableDescriptor, VariablesContainer> Variables;
  typedef SimpleDict<EvaluableDescriptor, EvaluablesContainer> Evaluables;
  typedef SimpleDict<TransformationDescriptor,
                     TransformationsContainer> Transformations;
  GNAObject()
    : ParametrizedTypes::Base(), TransformationTypes::Base(),
      variables(ParametrizedTypes::Base::m_entries),
      evaluables(m_eventries),
      transformations(TransformationTypes::Base::m_entries)
    { }
  GNAObject(const GNAObject &other)
    : ParametrizedTypes::Base(other),
      TransformationTypes::Base(other),
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
    : ParametrizedTypes::Base(), TransformationTypes::Base(1),
      variables(ParametrizedTypes::Base::m_entries),
      evaluables(m_eventries),
      transformations(TransformationTypes::Base::m_entries)
  { }

  typedef TransformationTypes::Args Args;
  typedef TransformationTypes::Rets Rets;
  typedef TransformationTypes::Atypes Atypes;
  typedef TransformationTypes::Rtypes Rtypes;
  typedef TransformationTypes::Function Function;
  typedef TransformationTypes::TypesFunction TypesFunction;
  typedef TransformationTypes::Entry Entry;
  typedef TransformationTypes::Accessor Accessor;
  typedef TransformationTypes::Handle Handle;
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
