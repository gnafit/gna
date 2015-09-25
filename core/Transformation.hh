#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include <string>

#include "TObject.h"

#include "TransformationBase.hh"
#include "SimpleDict.hh"

#define TransformationDef(classname)                            \
  using Transformation<classname>::transformation_;

class InputDescriptor;
class OutputDescriptor;
class TransformationDescriptor: public TObject,
                                public TransformationTypes::Handle {
public:
  typedef TransformationTypes::Handle BaseClass;

  typedef TransformationTypes::SourcesContainer SourcesContainer;
  typedef SimpleDict<InputDescriptor, SourcesContainer> Inputs;

  typedef TransformationTypes::SinksContainer SinksContainer;
  typedef SimpleDict<OutputDescriptor, SinksContainer> Outputs;

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
  static InputDescriptor invalid(const std::string name);

  bool connect(const OutputDescriptor &out) const;

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
  static OutputDescriptor invalid(const std::string name);

  bool connect(const InputDescriptor &in) const;

  const std::string name;

  ClassDef(OutputDescriptor, 0);
};

inline bool InputDescriptor::connect(const OutputDescriptor &out) const {
  return BaseClass::connect(out);
}

inline bool OutputDescriptor::connect(const InputDescriptor &in) const {
  return BaseClass::connect(in);
}

#endif // TRANSFORMATION_H
