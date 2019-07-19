#pragma once

#include <string>

#include "GNAObject.hh"
#include "TypeClasses.hh"
using namespace TypeClasses;

namespace GNA{
  namespace GNAObjectTemplates{
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
    template<typename FloatType>
    class FillLikeT: public GNASingleObjectT<FloatType,FloatType>,
                     public TransformationBind<FillLikeT<FloatType>,FloatType,FloatType> {
    private:
      using GNAObject = GNAObjectT<FloatType,FloatType>;
      using FillLike = FillLikeT<FloatType>;
      using typename GNAObject::FunctionArgs;
    public:
      /**
       * @brief Constructor.
       * @param value -- the value to be written to an array.
       */
      FillLikeT(FloatType value)
        : m_value(value)
      {
        this->transformation_("fill")                          // Initialize tranformation 'fill':
          .input("a")                                          // - with single input 'a'
          .output("a")                                         // - and single output 'a'
          .types(new PassTypeT<FloatType>(0,{0,-1}))           // - the shape of the output is taken from the input
          .func([](FillLike *obj, FunctionArgs& fargs) {       // - The implementation function:
              auto& rets=fargs.rets;
              rets[0].x.setConstant(obj->m_value);             //   set each element of a single output to m_value.
              rets.untaint();
              rets.freeze();
            });
      }
    protected:
      double m_value;  /// The value to be writte to input array.
    };
  }
}

using FillLike = GNA::GNAObjectTemplates::FillLikeT<double>;
