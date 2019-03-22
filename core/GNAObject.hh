#pragma once

#include <boost/noncopyable.hpp>
#include "ParametrizedBase.hh"
#include "VariableDescriptor.hh"
#include "EvaluableDescriptor.hh"
#include "TransformationDescriptor.hh"
#include "TransformationBase.hh"
#include "TransformationBind.hh"
#include "GPUFunctionArgs.hh"

template <typename SourceFloatType,typename SinkFloatType> class GNAObjectT;

template <>
class GNAObjectT<void,void>: public boost::noncopyable {
protected:
  GNAObjectT() = default;
  virtual ~GNAObjectT(){}
};

template <typename SourceFloatType,typename SinkFloatType>
class GNAObjectT: public GNAObjectT<void,void>,
                  public virtual TransformationTypes::BaseT<SourceFloatType,SinkFloatType>,
                  public virtual ParametrizedTypes::ParametrizedBase {
public:
  using VariablesContainer = ParametrizedTypes::VariablesContainer;
  using EvaluablesContainer = ParametrizedTypes::EvaluablesContainer;
  using Variables = SimpleDict<VariableDescriptor, VariablesContainer>;
  using Evaluables = SimpleDict<EvaluableDescriptor, EvaluablesContainer>;
  using TransformationBaseType = TransformationTypes::BaseT<SourceFloatType,SinkFloatType>;
  using TransformationsContainer = typename TransformationBaseType::EntryContainerType;
  using TransformationDescriptorType = TransformationDescriptorT<SourceFloatType,SinkFloatType>;
  using Transformations = SimpleDict<TransformationDescriptorType, TransformationsContainer>;
  using GNAObjectType = GNAObjectT<SourceFloatType,SinkFloatType>;

  GNAObjectT()
    : TransformationBaseType(),
      ParametrizedTypes::ParametrizedBase(),
      variables(ParametrizedTypes::ParametrizedBase::m_entries),
      evaluables(m_eventries),
      transformations(TransformationBaseType::m_entries)
    { }
  //GNAObjectT(const GNAObjectType &other)
    //: TransformationBaseType(other),
      //ParametrizedTypes::ParametrizedBase(other),
      //variables(ParametrizedTypes::ParametrizedBase::m_entries),
      //evaluables(m_eventries),
      //transformations(TransformationBaseType::m_entries)
    //{ }
  virtual ~GNAObjectT(){}

  void subscribe(taintflag flag) {
    ParametrizedTypes::ParametrizedBase::subscribe_(flag);
  }

  TransformationDescriptorType operator[](size_t idx) {
    return TransformationDescriptorType(TransformationBaseType::getEntry(idx));
  }

  TransformationDescriptorType operator[](const std::string &name) {
    return TransformationDescriptorType(TransformationBaseType::getEntry(name));
  }

  void dumpObj();

  void variablesBound();

  Variables variables;
  Evaluables evaluables;
  Transformations transformations;
protected:
  class SingleTransformation { };
  GNAObjectT(SingleTransformation)
    : TransformationBaseType(1),
      ParametrizedTypes::ParametrizedBase(),
      variables(ParametrizedTypes::ParametrizedBase::m_entries),
      evaluables(m_eventries),
      transformations(TransformationBaseType::m_entries)
  { }

  using Args = TransformationTypes::ArgsT<SourceFloatType,SinkFloatType>;
  using Rets = TransformationTypes::RetsT<SourceFloatType,SinkFloatType>;
  using Atypes = TransformationTypes::AtypesT<SourceFloatType,SinkFloatType>;
  using Rtypes = TransformationTypes::RtypesT<SourceFloatType,SinkFloatType>;
  using Entry = TransformationTypes::EntryT<SourceFloatType,SinkFloatType>;
  using Accessor = TransformationTypes::AccessorT<SourceFloatType,SinkFloatType>;
  using Handle = TransformationTypes::HandleT<SourceFloatType,SinkFloatType>;
  using OutputHandle = TransformationTypes::OutputHandleT<SinkFloatType>;
  using TransformationBaseType::t_;

  using OutputDescriptor = OutputDescriptorT<SourceFloatType,SinkFloatType>;
  using OutputDescriptors = typename OutputDescriptor::OutputDescriptors;
  using InputDescriptor = InputDescriptorT<SourceFloatType,SinkFloatType>;
  using SingleOutput = SingleOutputT<SinkFloatType>;

  using Function = TransformationTypes::FunctionT<SourceFloatType,SinkFloatType>;
  using TypesFunction = TransformationTypes::TypesFunctionT<SourceFloatType,SinkFloatType>;
  using StorageTypesFunctionArgs = TransformationTypes::StorageTypesFunctionArgsT<SourceFloatType,SinkFloatType>;
  using TypesFunctionArgs = TransformationTypes::TypesFunctionArgsT<SourceFloatType,SinkFloatType>;
  using FunctionArgs = TransformationTypes::FunctionArgsT<SourceFloatType,SinkFloatType>;
};

template<typename SourceFloatType, typename SinkFloatType>
class GNASingleObjectT: public GNAObjectT<SourceFloatType,SinkFloatType>,
                        public SingleOutputT<SinkFloatType> {
public:
  using GNAObjectType = GNAObjectT<SourceFloatType,SinkFloatType>;
  using GNASingleObjectType = GNASingleObjectT<SourceFloatType,SinkFloatType>;
  using OutputHandle = TransformationTypes::OutputHandleT<SinkFloatType>;
  using SingleTransformation = typename GNAObjectType::SingleTransformation;

  GNASingleObjectT()
    : GNAObjectT<SourceFloatType,SinkFloatType>(SingleTransformation())
  { }
  //GNASingleObjectT(const GNASingleObjectType &other)
    //: GNAObjectT<SourceFloatType,SinkFloatType>(other)
  //{ }

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

using GNAObject = GNAObjectT<double,double>;
using GNASingleObject = GNASingleObjectT<double,double>;

namespace GNA{
  std::vector<std::string> provided_precisions();
}
