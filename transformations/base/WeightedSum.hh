#pragma once

#include <string>
#include <vector>

#include "GNAObject.hh"


namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class WeightedSumT: public GNASingleObjectT<FloatType,FloatType>,
                       public TransformationBind<WeightedSumT<FloatType>, FloatType, FloatType> {
    
    private:
      using BaseClass = GNASingleObjectT<FloatType,FloatType>;
    public:
      using typename BaseClass::FunctionArgs;
      using typename BaseClass::TypesFunctionArgs;

      WeightedSumT(const std::vector<std::string> &labels);
      WeightedSumT(const std::vector<std::string> &weights, const OutputDescriptor::OutputDescriptors& outputs);
      WeightedSumT(const std::vector<std::string> &weights, const std::vector<std::string> &inputs);
      WeightedSumT(double fillvalue, const std::vector<std::string> &weights, const std::vector<std::string> &inputs);
    
    protected:
      WeightedSumT(bool use_fillvalue, const std::vector<std::string> &weights, const std::vector<std::string> &inputs);
    
      void sum(FunctionArgs& fargs);
      void sum_ongpu(FunctionArgs& fargs);
      void sumFill(FunctionArgs& fargs);
      void sumFill_ongpu(FunctionArgs& fargs);

      
      std::vector<variable<FloatType>> m_vars;
    
      FloatType m_fillvalue;
    };
  }
}

using WeightedSum = GNA::GNAObjectTemplates::WeightedSumT<double>;
