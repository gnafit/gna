#include "EntryHandle.hh"
#include "SingleOutput.hh"
#include "TransformationFunctionArgs.hh"
#include "GPUFunctionArgs.hh"

#include <algorithm>

using TransformationTypes::EntryT;
using TransformationTypes::HandleT;
using TransformationTypes::InputHandleT;
using TransformationTypes::OutputHandleT;

/**
 * @brief Get vector of inputs.
 *
 * The method creates a new vector and fills it with InputHandle instances
 * for each Entry's Source.
 *
 * @return new vector of inputs.
 */
template<typename SourceFloatType, typename SinkFloatType>
std::vector<InputHandleT<SourceFloatType>> HandleT<SourceFloatType,SinkFloatType>::inputs() const {
  std::vector<InputHandleT<SourceFloatType>> ret;
  auto &sources = m_entry->sources;
  std::transform(sources.begin(), sources.end(), std::back_inserter(ret),
                 [](SourceT<SourceFloatType> &s) { return InputHandleT<SourceFloatType>(s); });
  return ret;
}

/**
 * @brief Get vector of outputs.
 *
 * The method creates a new vector and fills it with OutputHandle instances
 * for each Entry's Sink.
 *
 * @return new vector of outputs.
 */
template<typename SourceFloatType, typename SinkFloatType>
std::vector<OutputHandleT<SinkFloatType>> HandleT<SourceFloatType,SinkFloatType>::outputs() const {
  std::vector<OutputHandleT<SinkFloatType>> ret;
  auto &sinks = m_entry->sinks;
  std::transform(sinks.begin(), sinks.end(), std::back_inserter(ret),
                 [](SinkT<SinkFloatType> &s) { return OutputHandleT<SinkFloatType>(s); });
  return ret;
}

/**
 * @brief Create a new input and connect to the SingleOutput transformation.
 *
 * New input name is copied from the output name.
 *
 * @param output -- SingleOutput transformation.
 * @return InputHandle for the new input.
 */
template<typename SourceFloatType, typename SinkFloatType>
InputHandleT<SourceFloatType> HandleT<SourceFloatType,SinkFloatType>::input(SingleOutputT<SourceFloatType> &output) {
  auto outhandle = output.single();
  auto inp = m_entry->addSource(outhandle.name());
  outhandle >> inp;
  return inp;
}

template<typename SourceFloatType, typename SinkFloatType>
void maplastinput(EntryT<SourceFloatType,SinkFloatType>* entry, int mapto){
  auto size=static_cast<int>(entry->sinks.size());
  if(mapto<0){
    mapto=size+mapto;
  }
  assert(mapto>=0 && mapto<size);
  auto& mapping = entry->mapping;
  if(mapping.size() < entry->sources.size()){
    mapping.resize(entry->sources.size());
  }
  //printf("map %zu -> %i\n", mapping.size(), mapto);
  mapping.back() = static_cast<size_t>(mapto);
}

/**
 * @brief Add named input.
 * @param name -- Source name.
 * @return InputHandle for the newly created Source.
 */
template<typename SourceFloatType, typename SinkFloatType>
InputHandleT<SourceFloatType> HandleT<SourceFloatType,SinkFloatType>::input(const std::string &name) {
  return m_entry->addSource(name);
}

/**
 * @brief Add named input.
 * @param name -- Source name.
 * @return InputHandle for the newly created Source.
 */
template<typename SourceFloatType, typename SinkFloatType>
InputHandleT<SourceFloatType> HandleT<SourceFloatType,SinkFloatType>::input(const std::string &name, int mapto) {
  auto inp=m_entry->addSource(name);
  maplastinput(m_entry, mapto);
  return inp;
}

/**
 * @brief Create a new input and connect to the SingleOutput transformation.
 *
 * New input name is copied from the output name.
 *
 * @param name   -- new input name.
 * @param output -- SingleOutput transformation.
 * @return InputHandle for the new input.
 */
template<typename SourceFloatType, typename SinkFloatType>
InputHandleT<SourceFloatType> HandleT<SourceFloatType,SinkFloatType>::input(const std::string& name, SingleOutputT<SourceFloatType> &output) {
  auto outhandle = output.single();
  auto inp = m_entry->addSource(name);
  outhandle >> inp;
  return inp;
}

/**
 * @brief Create a new input and connect to the SingleOutput transformation.
 *
 * New input name is copied from the output name.
 *
 * @param output -- SingleOutput transformation.
 * @param mapto -- number of the output the input corresponds to.
 * @return InputHandle for the new input.
 */
template<typename SourceFloatType, typename SinkFloatType>
InputHandleT<SourceFloatType> HandleT<SourceFloatType,SinkFloatType>::input(SingleOutputT<SourceFloatType> &output, int mapto) {
  auto ret = input(output);
  maplastinput(m_entry, mapto);
  return ret;
}

/**
 * @brief Create a new input and connect to the SingleOutput transformation.
 *
 * New input name is copied from the output name.
 *
 * @param name   -- new input name.
 * @param output -- SingleOutput transformation.
 * @param mapto -- number of the output the input corresponds to.
 * @return InputHandle for the new input.
 */
template<typename SourceFloatType, typename SinkFloatType>
InputHandleT<SourceFloatType> HandleT<SourceFloatType,SinkFloatType>::input(const std::string& name, SingleOutputT<SourceFloatType> &output, int mapto) {
  auto ret = input(name, output);
  maplastinput(m_entry, mapto);
  return ret;
}

/**
 * @brief Create a new output with a same name as SingleOutput's output.
 *
 * @param out -- SingleOutput transformation.
 * @return OutputHandle for the new output.
 */
template<typename SourceFloatType, typename SinkFloatType>
OutputHandleT<SinkFloatType> HandleT<SourceFloatType,SinkFloatType>::output(SingleOutputT<SourceFloatType> &out) {
  return output(out.single().name());
}

/**
 * @brief Assign variables to the transformation.
 *
 * @param out -- SingleOutput transformation.
 * @return OutputHandle for the new output.
 */
template<typename SourceFloatType, typename SinkFloatType>
void HandleT<SourceFloatType,SinkFloatType>::readVariables(ParametrizedTypes::ParametrizedBase* parbase){
  auto& gpu=m_entry->functionargs->gpu;
  std::cout << "TMPLOG READ VAR gpu" << std::endl;
  if(gpu){
  std::cout << "TMPLOG READ VAR gpu2" << std::endl;
    gpu->readVariables(parbase);
  }
}

/**
 * @brief Print Entry's Sink and Source instances and their connection status.
 *
 * The data is printed to the stderr.
 */
template<typename SourceFloatType, typename SinkFloatType>
void HandleT<SourceFloatType,SinkFloatType>::dumpObj() const {
  std::cerr << m_entry->name;
  std::cerr << std::endl;
  std::cerr << "    sources (" << m_entry->sources.size() << "):" << std::endl;
  int i = 0;
  for (auto &s: m_entry->sources) {
    std::cerr << "      " << i++ << ": " << s.name << ", ";
    if (s.sink) {
      std::cerr << "connected to ";
      std::cerr << s.sink->entry->name << "/" << s.sink->name << ", ";
      std::cerr << "type: ";
      s.sink->data->type.dump();
    } else {
      std::cerr << "not connected" << std::endl;
    }
  }
  std::cerr << "    sinks (" << m_entry->sinks.size() << "):" << std::endl;
  i = 0;
  for (auto &s: m_entry->sinks) {
    std::cerr << "      " << i++ << ": " << s.name << ", ";
    std::cerr << s.sources.size() << " consumers";
    std::cerr << ", type: ";
    s.data->type.dump();
  }
}

template class TransformationTypes::HandleT<double,double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::HandleT<float,float>;
#endif
