#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include <string>
#include <iostream>

#include "TransformationBase.hh"
#include "SimpleDict.hh"

class GNASingleObject;

class InputDescriptor;
class OutputDescriptor;

class TransformationDescriptor: public TransformationTypes::Handle,
                                public SingleOutput {
public:
  typedef TransformationTypes::Handle BaseClass;

  typedef TransformationTypes::SourcesContainer SourcesContainer;
  typedef SimpleDict<InputDescriptor, SourcesContainer> InputsBase;
  class Inputs;

  typedef TransformationTypes::SinksContainer SinksContainer;
  typedef SimpleDict<OutputDescriptor, SinksContainer> OutputsBase;
  class Outputs;

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

    TransformationTypes::OutputHandle single() override;
    TransformationTypes::OutputHandle single() const;
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

  TransformationTypes::OutputHandle single() override;
};

class InputDescriptor: public TransformationTypes::InputHandle {
public:
  typedef TransformationTypes::InputHandle BaseClass;

  InputDescriptor(const BaseClass &other)
    : BaseClass(other)
    { }
  InputDescriptor(const InputDescriptor &other)
    : InputDescriptor(BaseClass(other))
    { }
  InputDescriptor(TransformationTypes::Source &source)
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

  void connect(GNASingleObject &obj) const;
  void connect(const TransformationDescriptor &obj) const;
  void connect(const TransformationDescriptor::Outputs &outs) const;
  void connect(const OutputDescriptor &out) const;
};

class OutputDescriptor: public TransformationTypes::OutputHandle,
                        public SingleOutput {
public:
  typedef TransformationTypes::OutputHandle BaseClass;

  OutputDescriptor(const BaseClass &other)
    : BaseClass(other)
    { }
  OutputDescriptor(const OutputDescriptor &other)
    : OutputDescriptor(BaseClass(other))
    { }
  OutputDescriptor(TransformationTypes::Sink &sink)
    : OutputDescriptor(BaseClass(sink))
    { }
  static OutputDescriptor invalid(int index);
  static OutputDescriptor invalid(const std::string name);

  TransformationTypes::OutputHandle single() override { return *this; }
};

#endif // TRANSFORMATION_H
