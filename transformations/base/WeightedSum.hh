#pragma once

#include <string>
#include <vector>

#include "GNAObject.hh"

class WeightedSum: public GNASingleObject,
                   public TransformationBind<WeightedSum> {
public:
  WeightedSum(const std::vector<std::string> &labels, const std::vector<std::string> &weight_labels={});
  WeightedSum(double fillvalue, const std::vector<std::string> &labels, const std::vector<std::string> &weight_labels={});

protected:
  WeightedSum(bool use_fillvalue, const std::vector<std::string> &labels, const std::vector<std::string> &weight_labels);

  void sum(Args args, Rets rets);
  void sumFill(Args args, Rets rets);

  std::vector<variable<double>> m_vars;

  size_t m_common;
  double m_fillvalue;
};
