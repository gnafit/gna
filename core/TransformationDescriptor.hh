#pragma once

#include <string>
#include <iostream>

#include "EntryHandle.hh"
#include "SingleOutput.hh"
#include "SimpleDict.hh"

template<typename SourceFloatType, typename SinkFloatType> class OutputDescriptorT;
template<typename SourceFloatType, typename SinkFloatType> class InputDescriptorT;

template<typename SourceFloatType, typename SinkFloatType>
class GNASingleObjectT;

template<typename SourceFloatType, typename SinkFloatType>
class TransformationDescriptorT: public TransformationTypes::HandleT<SourceFloatType,SinkFloatType>,
                                 public SingleOutputT<SinkFloatType> {
public:
  using TransformationDescriptorType = TransformationDescriptorT<SourceFloatType,SinkFloatType>;
  using BaseClass = TransformationTypes::HandleT<SourceFloatType,SinkFloatType>;
  using EntryType = TransformationTypes::EntryT<SourceFloatType,SinkFloatType>;

  using SourcesContainer = TransformationTypes::SourcesContainerT<SourceFloatType>;
  using InputsBase = SimpleDict<InputDescriptorT<SourceFloatType,SinkFloatType>, SourcesContainer>;
  class Inputs;

  using SinksContainer = TransformationTypes::SinksContainerT<SinkFloatType>;
  using OutputsBase = SimpleDict<OutputDescriptorT<SourceFloatType,SinkFloatType>, SinksContainer>;
  class Outputs;

  using OutputHandle = TransformationTypes::OutputHandleT<SinkFloatType>;

  using GNASingleObjectType = GNASingleObjectT<SourceFloatType,SinkFloatType>;

  using BaseClass::m_entry;

  class Inputs: public InputsBase {
  public:
    Inputs(SourcesContainer &container)
      : InputsBase(container) { }
    void operator()(const Outputs &other) const;
    void operator()(const TransformationDescriptorType &other) const;
    void operator()(GNASingleObjectType &obj) const;
  };

  class Outputs: public OutputsBase,
                 public SingleOutputT<SinkFloatType> {
  public:
    Outputs(SinksContainer &container)
      : OutputsBase(container) { }

    OutputHandle single() override;
    OutputHandle single() const;
  };

  TransformationDescriptorT(const BaseClass &other)
    : TransformationTypes::HandleT<SourceFloatType,SinkFloatType>(other),
      inputs(m_entry->sources),
      outputs(m_entry->sinks)
    { }
  TransformationDescriptorT(const TransformationDescriptorType &other)
    : TransformationDescriptorT(BaseClass(other))
    { }
  TransformationDescriptorT(EntryType &entry)
    : TransformationDescriptorT(BaseClass(entry))
    { }
  static TransformationDescriptorType invalid(int index);
  static TransformationDescriptorType invalid(const std::string name);

  const Inputs inputs;
  const Outputs outputs;

  OutputHandle single() override;
};

template<typename SourceFloatType, typename SinkFloatType>
class InputDescriptorT: public TransformationTypes::InputHandleT<SourceFloatType> {
public:
  using InputDescriptorType          = InputDescriptorT<SourceFloatType,SinkFloatType>;
  using OutputDescriptorType         = OutputDescriptorT<SourceFloatType,SinkFloatType>;
  using BaseClass                    = TransformationTypes::InputHandleT<SourceFloatType>;
  using OutputHandleType             = TransformationTypes::OutputHandleT<SinkFloatType>;
  using GNASingleObjectType          = GNASingleObjectT<SourceFloatType,SinkFloatType>;
  using TransformationDescriptorType = TransformationDescriptorT<SourceFloatType,SinkFloatType>;
  using OutputsType                  = typename TransformationDescriptorType::Outputs;

  InputDescriptorT(const BaseClass &other)
    : BaseClass(other)
    { }
  InputDescriptorT(const InputDescriptorType &other)
    : InputDescriptorType(BaseClass(other))
    { }
  InputDescriptorT(TransformationTypes::SourceT<SourceFloatType> &source)
    : InputDescriptorType(BaseClass(source))
    { }
  static InputDescriptorType invalid(int index);
  static InputDescriptorType invalid(const std::string name);

  void operator()(GNASingleObjectType &obj) const {
    connect(obj);
  }
  void operator()(const TransformationDescriptorType &obj) const {
    connect(obj);
  }
  void operator()(const OutputsType &outs) const {
    connect(outs);
  }
  void operator()(const OutputDescriptorType &out) const {
    connect(out);
  }
  void operator()(const OutputHandleType &out) const {
    connect(out);
  }

  void connect(GNASingleObjectType &obj) const;
  void connect(const TransformationDescriptorType &obj) const;
  void connect(const OutputsType &outs) const;
  void connect(const OutputDescriptorType &out) const;
  void connect(const OutputHandleType &out) const;

  inline const OutputDescriptorType output() const;
};

template<typename SourceFloatType, typename SinkFloatType>
class OutputDescriptorT: public TransformationTypes::OutputHandleT<SinkFloatType>,
                         public SingleOutputT<SinkFloatType> {
public:
  using InputDescriptorType  = InputDescriptorT<SourceFloatType,SinkFloatType>;
  using OutputDescriptorType = OutputDescriptorT<SourceFloatType,SinkFloatType>;
  using BaseClass            = TransformationTypes::OutputHandleT<SinkFloatType>;
  using GNASingleObjectType  = GNASingleObjectT<SourceFloatType,SinkFloatType>;
  using BaseClass::data;

  OutputDescriptorT(const BaseClass &other)
    : BaseClass(other)
    { }
  OutputDescriptorT(const OutputDescriptorType &other)
    : OutputDescriptorType(BaseClass(other))
    { }
  OutputDescriptorT(TransformationTypes::SinkT<SinkFloatType> &sink)
    : OutputDescriptorType(BaseClass(sink))
    { }
  static OutputDescriptorType invalid(int index);
  static OutputDescriptorType invalid(const std::string name);

  BaseClass single() override { return *this; }

  typedef std::vector<OutputDescriptorType*> OutputDescriptors;
};

template<typename SourceFloatType, typename SinkFloatType>
const OutputDescriptorT<SourceFloatType,SinkFloatType> InputDescriptorT<SourceFloatType,SinkFloatType>::output() const {
  return OutputDescriptorT<SourceFloatType,SinkFloatType>(BaseClass::output());
}

using TransformationDescriptor = TransformationDescriptorT<double,double>;
using OutputDescriptor = OutputDescriptorT<double,double>;
using InputDescriptor = InputDescriptorT<double,double>;

