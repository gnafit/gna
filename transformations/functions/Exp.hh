#pragma once

#include "GNAObject.hh"
#include "Statistic.hh"

namespace GNA{
  namespace GNAObjectTemplates{
    /**
     * @brief Transformation to calculate the value of Exp(x)
     *
     * Inputs:
     *   - exp.points
     *   - exp.result
     *
     * @author Maxim Gonchar
     * @date 27.02.2018
     */
    template<typename FloatType>
    class ExpT: public GNASingleObjectT<FloatType,FloatType>,
               public TransformationBind<ExpT<FloatType>, FloatType, FloatType> {

    private:
      using BaseClass = GNAObjectT<FloatType,FloatType>;
    public:
      using typename BaseClass::FunctionArgs;
      using typename BaseClass::TypesFunctionArgs;
      ExpT();                               ///< Constructor.

      void calculate(FunctionArgs& fargs); ///< Calculate the value of function.
#ifdef GNA_CUDA_SUPPORT
      void calc_gpu(FunctionArgs& fargs); ///< Calculate the value of function on GPU.
#endif
    protected:
    };
  }
}

using Exp = GNA::GNAObjectTemplates::ExpT<double>;
