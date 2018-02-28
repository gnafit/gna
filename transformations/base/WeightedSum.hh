#pragma once

#include <string>
#include <vector>

#include "GNAObject.hh"

class WeightedSum: public GNASingleObject,
                   public TransformationBind<WeightedSum> {
public:
  WeightedSum(const std::vector<std::string> &labels, const std::vector<std::string> &weight_labels={});
protected:
  void sumEq(Args args, Rets rets);
  void sumArr(Args args, Rets rets);
  void sumVal(Args args, Rets rets);
  std::vector<variable<double>> m_vars;
};
