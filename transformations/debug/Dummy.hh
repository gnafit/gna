#pragma once

#include <stdio.h>
#include "GNAObject.hh"
#include "TypesFunctions.hh"
#include <vector>

/**
 * @brief Dummy transformation for testing pupropses.
 *
 * Does nothing. May have any number of inputs/outputs and use variables.
 *
 * @author Maxim Gonchar
 * @date 2017.05
 */
class Dummy: public GNASingleObject,
             public TransformationBind<Dummy> {
public:
  /**
   * @brief Constructor.
   * @param shape - size of each output.
   * @param label - transformation label.
   * @param labels - variables to bind.
   */
  Dummy(size_t shape, const char* label, const std::vector<std::string> &labels={});

  /** @brief Add an input by name and leave unconnected. */
  InputDescriptor add_input(const std::string& name){
    auto trans = transformations.back();
    auto input = trans.input(name);
    return InputDescriptor(input);
  }

  /** @brief Add an input. */
  OutputDescriptor add_input(SingleOutput& output, const std::string& name){
    auto out=output.single();
    auto input=add_input(name.size() ? name : out.name());
    output.single() >> input;
    return OutputDescriptor(transformations.back().outputs.back());
  }

  /** @brief Add an output by name */
  OutputDescriptor add_output(const std::string& name){
    auto trans = transformations.back();
    auto output = trans.output(name);
    trans.updateTypes();
    return OutputDescriptor(output);
  }

  void dummy_fcn(FunctionArgs& fargs);
  void dummy_gpuargs_h(FunctionArgs& fargs);

  std::vector<variable<double>> m_vars;
};
