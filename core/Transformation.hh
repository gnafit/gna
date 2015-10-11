#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include <string>
#include <iostream>
#include "TObject.h"

#include "TransformationBase.hh"
#include "SimpleDict.hh"

#define TransformationDef(classname)                            \
  using Transformation<classname>::transformation_;

class GNASingleObject;

class InputDescriptor;
class OutputDescriptor;
class TransformationDescriptor: public TObject,
                                public TransformationTypes::Handle {
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

    ClassDef(Inputs, 0);
  };

  class Outputs: public OutputsBase {
  public:
    Outputs(SinksContainer &container)
      : OutputsBase(container) { }

    OutputDescriptor single() const;

    ClassDef(Outputs, 0);
  };

  TransformationDescriptor(const BaseClass &other)
    : Handle(other), name(BaseClass::name()),
      inputs(m_entry->sources),
      outputs(m_entry->sinks)
    { }
  TransformationDescriptor(const TransformationDescriptor &other)
    : TransformationDescriptor(BaseClass(other))
    { }
  TransformationDescriptor(TransformationTypes::Entry &entry)
    : TransformationDescriptor(BaseClass(entry))
    { }

  const std::string name;
  const Inputs inputs;
  const Outputs outputs;

  ClassDef(TransformationDescriptor, 0);
};

class InputDescriptor: public TObject,
                       public TransformationTypes::InputHandle {
public:
  typedef TransformationTypes::InputHandle BaseClass;

  InputDescriptor(const BaseClass &other)
    : BaseClass(other), name(BaseClass::name())
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

  const std::string name;

  ClassDef(InputDescriptor, 0);
};

class OutputDescriptor: public TObject,
                        public TransformationTypes::OutputHandle {
public:
  typedef TransformationTypes::OutputHandle BaseClass;

  OutputDescriptor(const BaseClass &other)
    : BaseClass(other), name(BaseClass::name())
    { }
  OutputDescriptor(const OutputDescriptor &other)
    : OutputDescriptor(BaseClass(other))
    { }
  OutputDescriptor(TransformationTypes::Sink &sink)
    : OutputDescriptor(BaseClass(sink))
    { }
  static OutputDescriptor invalid(int index);
  static OutputDescriptor invalid(const std::string name);

  const std::string name;

  ClassDef(OutputDescriptor, 0);
};

#endif // TRANSFORMATION_H
