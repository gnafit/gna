#include <stdexcept>

#include <boost/format.hpp>
using boost::format;

#include "Transformation.hh"
#include "Exceptions.hh"
#include "GNAObject.hh"

using std::string;

using Input = InputDescriptor;
using Inputs = TransformationDescriptor::Inputs;
using Output = OutputDescriptor;
using Outputs = TransformationDescriptor::Outputs;

void connect(const Inputs &inputs, const Outputs &outputs) {
  if (inputs.size() != outputs.size()) {
    throw std::runtime_error("inconsistent sizes");
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    inputs.at(i).connect(outputs.at(i));
  }
}

TransformationTypes::OutputHandle Outputs::single() const {
  if (size() > 1) {
    throw std::runtime_error("too much outputs for one input");
  } else if (size() < 1) {
    throw std::runtime_error("no outputs");
  }
  return at(0);
}

TransformationTypes::OutputHandle Outputs::single() {
  return static_cast<const Outputs*>(this)->single();
}

TransformationTypes::OutputHandle TransformationDescriptor::single() {
  return outputs.single();
}

void connect(const Input &input, const Outputs &outputs) {
  input.connect(outputs.single());
}

void Inputs::operator()(const Outputs &outputs) const {
  connect(*this, outputs);
}

void Inputs::operator()(const TransformationDescriptor &other) const {
  connect(*this, other.outputs);
}

void Inputs::operator()(GNASingleObject &obj) const {
  connect(*this, obj[0].outputs);
}

void InputDescriptor::connect(GNASingleObject &obj) const {
  ::connect(*this, obj[0].outputs);
}

void InputDescriptor::connect(const TransformationDescriptor &obj) const {
  ::connect(*this, obj.outputs);
}

void InputDescriptor::connect(const Outputs &outs) const {
  ::connect(*this, outs);
}

void InputDescriptor::connect(const OutputDescriptor &out) const {
  return BaseClass::connect(out);
}

Input InputDescriptor::invalid(int index) {
  throw IndexError(index, "input");
}

Input InputDescriptor::invalid(const std::string name) {
  throw KeyError(name, "input");
}

Output OutputDescriptor::invalid(int index) {
  throw IndexError(index, "output");
}

Output OutputDescriptor::invalid(const std::string name) {
  throw KeyError(name, "output");
}
