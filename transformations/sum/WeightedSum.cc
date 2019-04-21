#include "WeightedSum.hh"
#include "TypesFunctions.hh"
#include "config_vars.h"
#ifdef GNA_CUDA_SUPPORT
#include "cuElementary.hh"                             
#include "DataLocation.hh"
#endif

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    WeightedSumT<FloatType>::WeightedSumT(const std::vector<std::string> &weights)
      : WeightedSumT(false, weights, weights){ }
    
    template<typename FloatType>
    WeightedSumT<FloatType>::WeightedSumT(const std::vector<std::string> &weights, const std::vector<std::string> &inputs)
      : WeightedSumT(false, weights, inputs){ }
    
    template<typename FloatType>
    WeightedSumT<FloatType>::WeightedSumT(double fillvalue, const std::vector<std::string> &weights, const std::vector<std::string> &inputs)
      : WeightedSumT(true, weights, inputs) { m_fillvalue=fillvalue; }
    
    template<typename FloatType>
    WeightedSumT<FloatType>::WeightedSumT(const std::vector<std::string> &weights, const OutputDescriptor::OutputDescriptors& outputs)
      : WeightedSumT(false, weights, weights)
    {
      const auto &trans  = this->transformations.front();
      auto &inputs = trans.inputs;
    
      if(inputs.size()!=outputs.size()){
        throw std::runtime_error("WeightedSum got inconsistent inputs and outputs");
      }
    
      for (size_t i = 0; i < outputs.size(); ++i) {
        inputs[i](*outputs[i]);
      }
    }
    
    template<typename FloatType>
    WeightedSumT<FloatType>::WeightedSumT(bool use_fillvalue, const std::vector<std::string> &weights, const std::vector<std::string> &inputs) {
      if (inputs.empty()) {
        return;
      }
    
      if(weights.size()>inputs.size()){
        throw std::runtime_error("WeightedSum should have at least as much inputs as the number of weights");
      }
    
      auto sum = this->transformation_("sum")
        .output("sum")
        .label("wsum")
        .types(TypesFunctions::ifSame, TypesFunctions::pass<0>)
        ;
    
      if( use_fillvalue ){
        sum.func(&WeightedSumT<FloatType>::sumFill)
    #ifdef GNA_CUDA_SUPPORT
           .func("gpu", &WeightedSumT<FloatType>::sumFill_ongpu, DataLocation::Device);
    #endif
      }
      else{
        sum.func(&WeightedSumT<FloatType>::sum)
    #ifdef GNA_CUDA_SUPPORT
           .func("gpu", &WeightedSumT<FloatType>::sum_ongpu, DataLocation::Device);
    #endif
      }
    
      m_vars.resize(weights.size());
      for (size_t i = 0; i < m_vars.size(); ++i) {
        this->variable_(&m_vars[i], weights[i].data());
      }
      for (auto& label: inputs) {
        sum.input(label);
      }
    }
    
    template<typename FloatType>
    void WeightedSumT<FloatType>::sum(FunctionArgs& fargs){
        auto& args=fargs.args;
        auto& ret=fargs.rets[0].x;
        ret = m_vars[0]*args[0].x;
        size_t i = 1;
        for (; i < m_vars.size(); ++i) {
          ret += m_vars[i]*args[i].x;
        }
        for (; i < args.size(); ++i) {
          ret += args[i].x;
        }
    }
    
    template<typename FloatType>
    void WeightedSumT<FloatType>::sumFill(FunctionArgs& fargs){
        auto& args=fargs.args;
        auto& ret=fargs.rets[0].x;
        ret = m_fillvalue;
        size_t i = 0;
        for (; i < m_vars.size(); ++i) {
          ret += m_vars[i]*args[i].x;
        }
        for (; i < args.size(); ++i) {
          ret += args[i].x;
        }
    }
    
    template<typename FloatType>
    void WeightedSumT<FloatType>::sum_ongpu(FunctionArgs& fargs) {
        fargs.args.touch();
        auto& gpuargs=fargs.gpu;
        //gpuargs->readVariables(m_vars);
        gpuargs->provideSignatureDevice();
        cuweightedsum(gpuargs->args, gpuargs->rets, gpuargs->vars, fargs.args[0].arr.size(), gpuargs->nargs, gpuargs->nvars);
        gpuargs->setAsDevice();
    }
    
    template<typename FloatType>
    void WeightedSumT<FloatType>::sumFill_ongpu(FunctionArgs& fargs) {
        fargs.args.touch();
        auto& gpuargs=fargs.gpu;
       // gpuargs->readVariables(m_vars);
        gpuargs->provideSignatureDevice();
        cuweightedsumfill(gpuargs->args, gpuargs->rets, gpuargs->vars, m_fillvalue, fargs.args[0].arr.size(), gpuargs->nargs, gpuargs->nvars);
        gpuargs->setAsDevice();
    }
  }
}

template class GNA::GNAObjectTemplates::WeightedSumT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::WeightedSumT<float>;
#endif
