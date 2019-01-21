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
  Dummy(size_t shape, const char* label, const std::vector<std::string> &labels={}){
    transformation_("dummy")
      .label(label)
      .types([shape](TypesFunctionArgs fargs) {
                for (size_t i = 0; i < fargs.rets.size(); ++i) {
                  fargs.rets[i] = DataType().points().shape(shape);
                }
              }
            )
    .func([](FunctionArgs& fargs){ fargs.args[0].x=1.0; });

    m_vars.resize(labels.size());
    for (size_t i = 0; i < m_vars.size(); ++i) {
      variable_(&m_vars[i], labels[i].data());
    }
  };

  /** @brief Add an input by name and leave unconnected. */
  InputDescriptor add_input(const char* name){
    return InputDescriptor(t_[0].input(name));
  }

  /** @brief Add an output by name */
  OutputDescriptor add_output(const char* name){
    auto ret = OutputDescriptor(t_[0].output(name));
    t_[0].updateTypes();
    return ret;
  }

  std::vector<variable<double>> m_vars;
};
