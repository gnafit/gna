#include "EntryHandle.hh"
#include "SingleOutput.hh"

#include <algorithm>

using TransformationTypes::Handle;
using TransformationTypes::InputHandle;
using TransformationTypes::OutputHandle;
using TransformationTypes::Sink;
using TransformationTypes::Source;

/**
 * @brief Get vector of inputs.
 *
 * The method creates a new vector and fills it with InputHandle instances
 * for each Entry's Source.
 *
 * @return new vector of inputs.
 */
std::vector<InputHandle> Handle::inputs() const {
  std::vector<InputHandle> ret;
  auto &sources = m_entry->sources;
  std::transform(sources.begin(), sources.end(), std::back_inserter(ret),
                 [](Source &s) { return InputHandle(s); });
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
std::vector<OutputHandle> Handle::outputs() const {
  std::vector<OutputHandle> ret;
  auto &sinks = m_entry->sinks;
  std::transform(sinks.begin(), sinks.end(), std::back_inserter(ret),
                 [](Sink &s) { return OutputHandle(s); });
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
InputHandle Handle::input(SingleOutput &output) {
  OutputHandle outhandle = output.single();
  InputHandle inp = m_entry->addSource(outhandle.name());
  inp.connect(outhandle);
  return inp;
}

/**
 * @brief Create a new output with a same name as SingleOutput's output.
 *
 * @param out -- SingleOutput transformation.
 * @return OutputHandle for the new output.
 */
OutputHandle Handle::output(SingleOutput &out) {
  return output(out.single().name());
}

/**
 * @brief Print Entry's Sink and Source instances and their connection status.
 *
 * The data is printed to the stderr.
 */
void Handle::dumpObj() const {
  std::cerr << m_entry->name;
  std::cerr << std::endl;
  std::cerr << "    sources (" << m_entry->sources.size() << "):" << std::endl;
  int i = 0;
  for (Source &s: m_entry->sources) {
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
  for (Sink &s: m_entry->sinks) {
    std::cerr << "      " << i++ << ": " << s.name << ", ";
    std::cerr << s.sources.size() << " consumers";
    std::cerr << ", type: ";
    s.data->type.dump();
  }
}
