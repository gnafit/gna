#include <stdexcept>

#include "TransformationDescriptor.hh"
#include "Exceptions.hh"
#include "GNAObject.hh"

using std::string;
using TransformationTypes::OutputHandleT;

template<typename SourceFloatType, typename SinkFloatType>
void connect(const typename TransformationDescriptorT<SourceFloatType,SinkFloatType>::Inputs &inputs,
             const typename TransformationDescriptorT<SourceFloatType,SinkFloatType>::Outputs &outputs) {
  if (inputs.size() != outputs.size()) {
    throw std::runtime_error("inconsistent sizes");
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    inputs.at(i).connect(outputs.at(i));
  }
}

template<typename SourceFloatType, typename SinkFloatType>
void connect(const InputDescriptorT<SourceFloatType,SinkFloatType> &input,
             const typename TransformationDescriptorT<SourceFloatType,SinkFloatType>::Outputs &outputs) {
  input.connect(outputs.single());
}

template<typename SourceFloatType, typename SinkFloatType>
OutputHandleT<SinkFloatType> TransformationDescriptorT<SourceFloatType,SinkFloatType>::Outputs::single() const {
  if (size() > 1) {
    throw std::runtime_error("too much outputs for one input");
  } else if (size() < 1) {
    throw std::runtime_error("no outputs");
  }
  return at(0);
}

template<typename SourceFloatType, typename SinkFloatType>
OutputHandleT<SinkFloatType> TransformationDescriptorT<SourceFloatType,SinkFloatType>::Outputs::single() {
  using Outputs = typename TransformationDescriptorT<SourceFloatType,SinkFloatType>::Outputs;
  return static_cast<const Outputs*>(this)->single();
}

template<typename SourceFloatType, typename SinkFloatType>
OutputHandleT<SinkFloatType> TransformationDescriptorT<SourceFloatType,SinkFloatType>::single() {
  return outputs.single();
}

template<typename SourceFloatType, typename SinkFloatType>
void TransformationDescriptorT<SourceFloatType,SinkFloatType>::Inputs::operator()(const TransformationDescriptorT<SourceFloatType,SinkFloatType>::Outputs &outputs) const {
  ::connect<SourceFloatType,SinkFloatType>(*this, outputs);
}

template<typename SourceFloatType, typename SinkFloatType>
void TransformationDescriptorT<SourceFloatType,SinkFloatType>::Inputs::operator()(const TransformationDescriptorT<SourceFloatType,SinkFloatType> &other) const {
  ::connect<SourceFloatType,SinkFloatType>(*this, other.outputs);
}

template<typename SourceFloatType, typename SinkFloatType>
void TransformationDescriptorT<SourceFloatType,SinkFloatType>::Inputs::operator()(GNASingleObjectT<SourceFloatType,SinkFloatType> &obj) const {
  ::connect<SourceFloatType,SinkFloatType>(*this, obj[0].outputs);
}

template<typename SourceFloatType, typename SinkFloatType>
void InputDescriptorT<SourceFloatType,SinkFloatType>::connect(GNASingleObjectT<SourceFloatType,SinkFloatType> &obj) const {
  ::connect<SourceFloatType,SinkFloatType>(*this, obj[0].outputs);
}

template<typename SourceFloatType, typename SinkFloatType>
void InputDescriptorT<SourceFloatType,SinkFloatType>::connect(const TransformationDescriptorT<SourceFloatType,SinkFloatType> &obj) const {
  ::connect<SourceFloatType,SinkFloatType>(*this, obj.outputs);
}

template<typename SourceFloatType, typename SinkFloatType>
void InputDescriptorT<SourceFloatType,SinkFloatType>::connect(const typename TransformationDescriptorT<SourceFloatType,SinkFloatType>::Outputs &outs) const {
  ::connect<SourceFloatType,SinkFloatType>(*this, outs);
}

template<typename SourceFloatType, typename SinkFloatType>
void InputDescriptorT<SourceFloatType,SinkFloatType>::connect(const OutputDescriptorT<SourceFloatType,SinkFloatType> &out) const {
  return BaseClass::connect(out);
}

template<typename SourceFloatType, typename SinkFloatType>
void InputDescriptorT<SourceFloatType,SinkFloatType>::connect(const OutputHandleT<SinkFloatType> &out) const {
  return BaseClass::connect(out);
}

template<typename SourceFloatType, typename SinkFloatType>
TransformationDescriptorT<SourceFloatType,SinkFloatType> TransformationDescriptorT<SourceFloatType,SinkFloatType>::invalid(int index) {
  throw IndexError(index, "input");
}

template<typename SourceFloatType, typename SinkFloatType>
TransformationDescriptorT<SourceFloatType,SinkFloatType> TransformationDescriptorT<SourceFloatType,SinkFloatType>::invalid(const std::string name) {
  throw KeyError(name, "input");
}

template<typename SourceFloatType, typename SinkFloatType>
InputDescriptorT<SourceFloatType,SinkFloatType> InputDescriptorT<SourceFloatType,SinkFloatType>::invalid(int index) {
  throw IndexError(index, "input");
}

template<typename SourceFloatType, typename SinkFloatType>
InputDescriptorT<SourceFloatType,SinkFloatType> InputDescriptorT<SourceFloatType,SinkFloatType>::invalid(const std::string name) {
  throw KeyError(name, "input");
}

template<typename SourceFloatType, typename SinkFloatType>
OutputDescriptorT<SourceFloatType,SinkFloatType> OutputDescriptorT<SourceFloatType,SinkFloatType>::invalid(int index) {
  throw IndexError(index, "output");
}

template<typename SourceFloatType, typename SinkFloatType>
OutputDescriptorT<SourceFloatType,SinkFloatType> OutputDescriptorT<SourceFloatType,SinkFloatType>::invalid(const std::string name) {
  throw KeyError(name, "output");
}


template class TransformationDescriptorT<double,double>;
template class InputDescriptorT<double,double>;
template class OutputDescriptorT<double,double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationDescriptorT<float,float>;
  template class InputDescriptorT<float,float>;
  template class OutputDescriptorT<float,float>;
#endif
