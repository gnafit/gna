#pragma once

#include <string>
#include <vector>

#include "GNAObject.hh"


namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class WeightedSumSqT: public GNASingleObjectT<FloatType,FloatType>,
                          public TransformationBind<WeightedSumSqT<FloatType>, FloatType, FloatType> {

    private:
      using BaseClass = GNAObjectT<FloatType,FloatType>;
    public:
      using typename BaseClass::FunctionArgs;
      using typename BaseClass::TypesFunctionArgs;
      using typename BaseClass::OutputDescriptor;

      WeightedSumSqT(const std::vector<std::string> &labels);
      WeightedSumSqT(const std::vector<std::string> &weights, const typename OutputDescriptor::OutputDescriptors& outputs);
      WeightedSumSqT(const std::vector<std::string> &weights, const std::vector<std::string> &inputs);
      WeightedSumSqT(double fillvalue, const std::vector<std::string> &weights, const std::vector<std::string> &inputs);

    protected:
      WeightedSumSqT(bool use_fillvalue, const std::vector<std::string> &weights, const std::vector<std::string> &inputs);

      void sum(FunctionArgs& fargs);
      void sumFill(FunctionArgs& fargs);

#ifdef GNA_CUDA_SUPPORT
      void sum_ongpu(FunctionArgs& fargs);
      void sumFill_ongpu(FunctionArgs& fargs);
#endif

      std::vector<variable<FloatType>> m_vars;

      FloatType m_fillvalue;
    };
  }
}

