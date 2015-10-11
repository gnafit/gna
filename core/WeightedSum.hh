#ifndef WEIGHTEDSUM_H
#define WEIGHTEDSUM_H

#include <string>
#include <vector>

#include "GNAObject.hh"

class WeightedSum: public GNAObject,
                   public Transformation<WeightedSum> {
public:
  TransformationDef(WeightedSum)
  WeightedSum(const std::vector<std::string> &labels);
protected:
  std::vector<variable<double>> m_vars;
};

#endif // WEIGHTEDSUM_H
