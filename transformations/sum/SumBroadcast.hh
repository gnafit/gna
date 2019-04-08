#pragma once

#include "GNAObject.hh"

/**
 * @brief Calculate the element-wise sum of the inputs.
 *
 * Outputs:
 *   - `sum.sum` -- the result of a sum.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
class SumBroadcast: public GNASingleObject,
                    public TransformationBind<SumBroadcast> {
public:
  SumBroadcast();                                                   ///< Constructor
  SumBroadcast(const OutputDescriptor::OutputDescriptors& outputs); ///< Construct SumBroadcast from vector of outputs

  /**
   * @brief Construct a SumBroadcast of two outputs
   * @param data1 -- first SingleOutput instance.
   * @param data2 -- second SingleOutput instance.
   */
  SumBroadcast(SingleOutput& data1, SingleOutput& data2) : SumBroadcast() {
    add(data1, data2);
  }

  /**
   * @brief Construct a SumBroadcast of three outputs
   * @param data1 -- first SingleOutput instance.
   * @param data2 -- second SingleOutput instance.
   * @param data3 -- third SingleOutput instance.
   */
  SumBroadcast(SingleOutput& data1, SingleOutput& data2, SingleOutput& data3) : SumBroadcast() {
    add(data1, data2, data3);
  }

  InputDescriptor add_input(const char* name);  ///< Add an input by name and leave unconnected.

  /** @brief Add an input by name and leave unconnected. */
  InputDescriptor add(const char* name){
    return add_input(name);
  }

  InputDescriptor add(SingleOutput &data); ///< Add an input and connect it to the output.

  /**
   * @brief Add two inputs and connect them to the outputs.
   *
   * @param data1 -- first SingleOutput instance.
   * @param data2 -- second SingleOutput instance.
   * @return InputDescriptor instance for the last created input.
   */
  InputDescriptor add(SingleOutput &data1, SingleOutput &data2){ add(data1); return add(data2); }

  /**
   * @brief Add three inputs and connect them to the outputs.
   *
   * @param data1 -- first SingleOutput instance.
   * @param data2 -- second SingleOutput instance.
   * @param data3 -- third SingleOutput instance.
   * @return InputDescriptor instance for the last created input.
   */
  InputDescriptor add(SingleOutput &data1, SingleOutput &data2, SingleOutput &data3){ add(data1); return add(data2, data3); }

};
