#include "WeightedSumSq.hh"
#include "TypeClasses.hh"
using namespace TypeClasses;

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    WeightedSumSqT<FloatType>::WeightedSumSqT(const std::vector<std::string> &weights)
      : WeightedSumSqT(false, weights, weights){ }

    template<typename FloatType>
    WeightedSumSqT<FloatType>::WeightedSumSqT(const std::vector<std::string> &weights, const std::vector<std::string> &inputs)
      : WeightedSumSqT(false, weights, inputs){ }

    template<typename FloatType>
    WeightedSumSqT<FloatType>::WeightedSumSqT(double fillvalue, const std::vector<std::string> &weights, const std::vector<std::string> &inputs)
      : WeightedSumSqT(true, weights, inputs) { m_fillvalue=fillvalue; }

    template<typename FloatType>
    WeightedSumSqT<FloatType>::WeightedSumSqT(const std::vector<std::string> &weights, const typename OutputDescriptor::OutputDescriptors& outputs)
      : WeightedSumSqT(false, weights, weights)
    {
      const auto &trans  = this->transformations.front();
      auto &inputs = trans.inputs;

      if(inputs.size()!=outputs.size()){
        throw std::runtime_error("WeightedSumSq got inconsistent inputs and outputs");
      }

      for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i] >> inputs[i];
      }
    }

    template<typename FloatType>
    WeightedSumSqT<FloatType>::WeightedSumSqT(bool use_fillvalue, const std::vector<std::string> &weights, const std::vector<std::string> &inputs) {
      if (inputs.empty()) {
        return;
      }

      if(weights.size()>inputs.size()){
        throw std::runtime_error("WeightedSumSq should have at least as much inputs as the number of weights");
      }

      auto sum = this->transformation_("sum")
        .output("sum")
        .label("wsum2")
        //.types(new PassTypeT<FloatType>(0,{0,-1})) // FIXME: set back!!!
        .types(new CheckSameTypesT<FloatType>({0,-1}, "shape"), new PassTypeT<FloatType>(0,{0,-1}))
        ;

      if( use_fillvalue ){
        sum.func(&WeightedSumSqT<FloatType>::sumFill)
    #ifdef GNA_CUDA_SUPPORT
           .func("gpu", &WeightedSumSqT<FloatType>::sumFill_ongpu, DataLocation::Device)
    #endif
        ;
      }
      else{
        sum.func(&WeightedSumSqT<FloatType>::sum)
    #ifdef GNA_CUDA_SUPPORT
           .func("gpu", &WeightedSumSqT<FloatType>::sum_ongpu, DataLocation::Device);
    #endif
        ;
      }
//   	sum.finalize();
      m_vars.resize(weights.size());
      for (size_t i = 0; i < m_vars.size(); ++i) {
        this->variable_(&m_vars[i], weights[i].data());
      }
      for (auto& label: inputs) {
        sum.input(label);
      }
    }

    template<typename FloatType>
    void WeightedSumSqT<FloatType>::sum(FunctionArgs& fargs){
        auto& args=fargs.args;
        auto& ret=fargs.rets[0].x;

        ret = pow(m_vars[0].value(), 2)*args[0].x.square();
        size_t i = 1;
        for (; i < m_vars.size(); ++i) {
          ret += (pow(m_vars[i].value(), 2))*args[i].x.square();
        }
        for (; i < args.size(); ++i) {
          ret += args[i].x.square();
        }
    }

    template<typename FloatType>
    void WeightedSumSqT<FloatType>::sumFill(FunctionArgs& fargs){
        auto& args=fargs.args;
        auto& ret=fargs.rets[0].x;
        ret = m_fillvalue*m_fillvalue;
        size_t i = 0;
        for (; i < m_vars.size(); ++i) {
          ret += pow(m_vars[i].value(), 2)*args[i].x.square();
        }
        for (; i < args.size(); ++i) {
          ret += args[i].x.square();
        }
    }

//#ifdef GNA_CUDA_SUPPORT
// Should be checked
    //template<typename FloatType>
    //void WeightedSumSqT<FloatType>::sum_ongpu(FunctionArgs& fargs) {
        //fargs.args.touch();
        //auto& gpuargs=fargs.gpu;
        //gpuargs->provideSignatureDevice();
        //cuweightedsum(gpuargs->args, gpuargs->rets, gpuargs->vars, fargs.args[0].arr.size(), gpuargs->nargs, gpuargs->nvars);
        ////gpuargs->setAsDevice();
    //}

    //template<typename FloatType>
    //void WeightedSumSqT<FloatType>::sumFill_ongpu(FunctionArgs& fargs) {
        //fargs.args.touch();
        //auto& gpuargs=fargs.gpu;
        //gpuargs->provideSignatureDevice();
        //cuweightedsumfill(gpuargs->args, gpuargs->rets, gpuargs->vars, m_fillvalue, fargs.args[0].arr.size(), gpuargs->nargs, gpuargs->nvars);
////        gpuargs->setAsDevice();
    //}
//#endif
  }
}

template class GNA::GNAObjectTemplates::WeightedSumSqT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::WeightedSumSqT<float>;
#endif
