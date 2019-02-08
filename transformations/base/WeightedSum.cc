#include "WeightedSum.hh"
#include "TypesFunctions.hh"
#include "config_vars.h"
#ifdef GNA_CUDA_SUPPORT
#include "cuElementary.hh"                             
#include "DataLocation.hh"
#endif


WeightedSum::WeightedSum(const std::vector<std::string> &weights)
  : WeightedSum(false, weights, weights){ }

WeightedSum::WeightedSum(const std::vector<std::string> &weights, const std::vector<std::string> &inputs)
  : WeightedSum(false, weights, inputs){ }

WeightedSum::WeightedSum(double fillvalue, const std::vector<std::string> &weights, const std::vector<std::string> &inputs)
  : WeightedSum(true, weights, inputs) { m_fillvalue=fillvalue; }

WeightedSum::WeightedSum(const std::vector<std::string> &weights, const OutputDescriptor::OutputDescriptors& outputs)
  : WeightedSum(false, weights, weights)
{
  const auto &trans  = transformations.front();
  auto &inputs = trans.inputs;

  if(inputs.size()!=outputs.size()){
    throw std::runtime_error("WeightedSum got inconsistent inputs and outputs");
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    inputs[i](*outputs[i]);
  }
}

WeightedSum::WeightedSum(bool use_fillvalue, const std::vector<std::string> &weights, const std::vector<std::string> &inputs) {
  if (inputs.empty()) {
    return;
  }

  if(weights.size()>inputs.size()){
    throw std::runtime_error("WeightedSum should have at least as much inputs as the number of weights");
  }

  auto sum = transformation_("sum")
    .output("sum")
    .label("wsum")
    .types(TypesFunctions::ifSame, TypesFunctions::pass<0>)
#ifdef GNA_CUDA_SUPPORT
    .func("gpu", &WeightedSum::sum_ongpu, DataLocation::Device)
#endif
    ;

  if( use_fillvalue ){
    sum.func(&WeightedSum::sumFill);
  }
  else{
    sum.func(&WeightedSum::sum);
  }

  m_vars.resize(weights.size());
  for (size_t i = 0; i < m_vars.size(); ++i) {
    variable_(&m_vars[i], weights[i].data());
  }
  for (auto& label: inputs) {
    sum.input(label);
  }
}

void WeightedSum::sum(FunctionArgs& fargs){
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

void WeightedSum::sumFill(FunctionArgs& fargs){
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

void WeightedSum::sum_ongpu(FunctionArgs& fargs) {
    fargs.args.touch();
    auto& gpuargs=fargs.gpu;
    gpuargs->readVariables(m_vars);
    gpuargs->provideSignatureDevice();
    cuweightedsum(gpuargs->args, gpuargs->rets, gpuargs->vars, fargs.args[0].arr.size(), gpuargs->nvars);
}
