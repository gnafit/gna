#ifndef GNAOBJECT_H
#define GNAOBJECT_H

#include "TObject.h"

#include "Parametrized.hh"
#include "Transformation.hh"

class GNAObject: public TObject,
                 public ParametrizedTypes::Base,
                 public TransformationTypes::Base {
public:
  typedef ParametrizedTypes::VariablesContainer VariablesContainer;
  typedef ParametrizedTypes::EvaluablesContainer EvaluablesContainer;
  typedef TransformationTypes::Container TransformationsContainer;
  typedef SimpleDict<VariableDescriptor, VariablesContainer> Variables;
  typedef SimpleDict<EvaluableDescriptor, EvaluablesContainer> Evaluables;
  typedef SimpleDict<TransformationDescriptor,
                     TransformationsContainer> Transformations;
  GNAObject()
    : TObject(), ParametrizedTypes::Base(), TransformationTypes::Base(),
      variables(ParametrizedTypes::Base::m_entries),
      evaluables(m_eventries),
      transformations(TransformationTypes::Base::m_entries)
    { }
  GNAObject(const GNAObject &other)
    : TObject(other), ParametrizedTypes::Base(other),
      TransformationTypes::Base(other),
      variables(ParametrizedTypes::Base::m_entries),
      evaluables(m_eventries),
      transformations(TransformationTypes::Base::m_entries)
    { }

  void subscribe(taintflag flag) {
    ParametrizedTypes::Base::subscribe_(flag);
  }

  TransformationDescriptor operator[](const std::string &name) {
    return TransformationDescriptor(TransformationTypes::Base::getEntry(name));
  }

  void dump();

  Variables variables;
  Evaluables evaluables;
  Transformations transformations;
protected:
  typedef TransformationTypes::Args Args;
  typedef TransformationTypes::Rets Rets;
  typedef TransformationTypes::Atypes Atypes;
  typedef TransformationTypes::Rtypes Rtypes;
  typedef TransformationTypes::Function Function;
  typedef TransformationTypes::TypesFunction TypesFunction;
  typedef TransformationTypes::Channel Channel;
  typedef TransformationTypes::Entry Entry;
  typedef TransformationTypes::Accessor Accessor;
  typedef TransformationTypes::Handle Handle;

  ClassDef(GNAObject, 0);
};

#endif // GNAOBJECT_H
