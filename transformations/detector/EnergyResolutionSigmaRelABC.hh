#pragma once

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class EnergyResolutionSigmaRelABC: public GNAObjectT<double, double>,
                                public TransformationBind<EnergyResolutionSigmaRelABC> {
public:
  EnergyResolutionSigmaRelABC(const std::vector<std::string>& pars);

private:
  void calcSigma(FunctionArgs& fargs);

  variable<double> m_a, m_b, m_c;
};
