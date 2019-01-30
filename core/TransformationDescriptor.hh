#pragma once

#include <string>
#include <iostream>

#include "EntryHandle.hh"
#include "SingleOutput.hh"
#include "SimpleDict.hh"

class GNASingleObject;

class InputDescriptor;
class OutputDescriptor;

class TransformationDescriptor: public TransformationTypes::Handle,
                                public SingleOutput {
public:
  using BaseClass = TransformationTypes::Handle;

  using SourcesContainer = TransformationTypes::SourcesContainerT<double>;
  using InputsBase = SimpleDict<InputDescriptor, SourcesContainer>;
  class Inputs;

  using SinksContainer = TransformationTypes::SinksContainerT<double>;
  using OutputsBase = SimpleDict<OutputDescriptor, SinksContainer>;
  class Outputs;

  using OutputHandle = TransformationTypes::OutputHandleT<double>;

  class Inputs: public InputsBase {
  public:
    Inputs(SourcesContainer &container)
      : InputsBase(container) { }
    void operator()(const Outputs &other) const;
    void operator()(const TransformationDescriptor &other) const;
    void operator()(GNASingleObject &obj) const;
  };

  class Outputs: public OutputsBase,
                 public SingleOutput {
  public:
    Outputs(SinksContainer &container)
      : OutputsBase(container) { }

    OutputHandle single() override;
    OutputHandle single() const;
  };

  TransformationDescriptor(const BaseClass &other)
    : Handle(other),
      inputs(m_entry->sources),
      outputs(m_entry->sinks)
    { }
  TransformationDescriptor(const TransformationDescriptor &other)
    : TransformationDescriptor(BaseClass(other))
    { }
  TransformationDescriptor(TransformationTypes::Entry &entry)
    : TransformationDescriptor(BaseClass(entry))
    { }
  static TransformationDescriptor invalid(int index);
  static TransformationDescriptor invalid(const std::string name);

  const Inputs inputs;
  const Outputs outputs;

  OutputHandle single() override;
};

class InputDescriptor: public TransformationTypes::InputHandleT<double> {
public:
  using BaseClass = TransformationTypes::InputHandleT<double>;
  using OutputHandle = TransformationTypes::OutputHandleT<double>;

  InputDescriptor(const BaseClass &other)
    : BaseClass(other)
    { }
  InputDescriptor(const InputDescriptor &other)
    : InputDescriptor(BaseClass(other))
    { }
  InputDescriptor(TransformationTypes::SourceT<double> &source)
    : InputDescriptor(BaseClass(source))
    { }
  static InputDescriptor invalid(int index);
  static InputDescriptor invalid(const std::string name);

  void operator()(GNASingleObject &obj) const {
    connect(obj);
  }
  void operator()(const TransformationDescriptor &obj) const {
    connect(obj);
  }
  void operator()(const TransformationDescriptor::Outputs &outs) const {
    connect(outs);
  }
  void operator()(const OutputDescriptor &out) const {
    connect(out);
  }
  void operator()(const OutputHandle &out) const {
    connect(out);
  }

  void connect(GNASingleObject &obj) const;
  void connect(const TransformationDescriptor &obj) const;
  void connect(const TransformationDescriptor::Outputs &outs) const;
  void connect(const OutputDescriptor &out) const;
  void connect(const OutputHandle &out) const;

  inline const OutputDescriptor output() const;
};

class OutputDescriptor: public TransformationTypes::OutputHandleT<double>,
                        public SingleOutput {
public:
  using BaseClass = TransformationTypes::OutputHandleT<double>;
  using OutputHandleT<double>::data;

  OutputDescriptor(const BaseClass &other)
    : BaseClass(other)
    { }
  OutputDescriptor(const OutputDescriptor &other)
    : OutputDescriptor(BaseClass(other))
    { }
  OutputDescriptor(TransformationTypes::SinkT<double> &sink)
    : OutputDescriptor(BaseClass(sink))
    { }
  static OutputDescriptor invalid(int index);
  static OutputDescriptor invalid(const std::string name);

  BaseClass single() override { return *this; }

  typedef std::vector<OutputDescriptor*> OutputDescriptors;
};

const OutputDescriptor InputDescriptor::output() const {
  return OutputDescriptor(BaseClass::output());
}
