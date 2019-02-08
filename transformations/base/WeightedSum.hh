#pragma once

#include <string>
#include <vector>

#include "GNAObject.hh"

class WeightedSum: public GNASingleObject,
                   public TransformationBind<WeightedSum> {
public:
  WeightedSum(const std::vector<std::string> &labels);
  WeightedSum(const std::vector<std::string> &weights, const OutputDescriptor::OutputDescriptors& outputs);
  WeightedSum(const std::vector<std::string> &weights, const std::vector<std::string> &inputs);
  WeightedSum(double fillvalue, const std::vector<std::string> &weights, const std::vector<std::string> &inputs);

protected:
  WeightedSum(bool use_fillvalue, const std::vector<std::string> &weights, const std::vector<std::string> &inputs);

  void sum(FunctionArgs& fargs);
  void sum_ongpu(FunctionArgs& fargs);
  void sumFill(FunctionArgs& fargs);
  void sumFill_ongpu(FunctionArgs& fargs);

  std::vector<variable<double>> m_vars;

  double m_fillvalue;
};
