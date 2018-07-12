#pragma once

#include <string>

#include "GNAObject.hh"
#include "TypesFunctions.hh"

/**
 * @brief FillLike transformation
 *
 * Fill an array with a number. The shape of an array is detemined by an input.
 *
 * Inputs and outputs:
 *   - `fill.inputs.a` -- the input array, needed to detemine the shape.
 *   - `fill.outputs.a` -- the output array.
 *
 * @author Dmitry Taychenachev
 * @date 02.2016
 */
class FillLike: public GNASingleObject,
                public TransformationBind<FillLike> {
public:
  /**
   * @brief Constructor.
   * @param value -- the value to be written to an array.
   */
  FillLike(double value)
    : m_value(value)
  {
    transformation_("fill")                                // Initialize tranformation 'fill':
      .input("a")                                          // - with single input 'a'
      .output("a")                                         // - and single output 'a'
      .types(TypesFunctions::passAll)                      // - the shape of the output is taken from the input
      .func([](FillLike *obj, Args /*args*/, Rets rets) {  // - The implementation function:
          rets[0].x.setConstant(obj->m_value);             //   set each element of a single output to m_value.
        });
  }
protected:
  double m_value;  /// The value to be writte to input array.
};
