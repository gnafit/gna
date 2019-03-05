#ifndef WEIGHTEDSUM_H
#define WEIGHTEDSUM_H

#include <string>
#include <vector>

#include "GNAObject.hh"

class WeightedSum: public GNASingleObject,
                   public Transformation<WeightedSum> {
public:
  WeightedSum(const std::vector<std::string> &labels, const std::vector<std::string> &weight_labels={});
protected:
  std::vector<variable<double>> m_vars;
};

#endif // WEIGHTEDSUM_H
