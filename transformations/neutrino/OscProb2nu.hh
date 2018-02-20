#pragma once

#include <Eigen/Dense>

#include "GNAObject.hh"
#include "OscillationVariables.hh"

class OscProb2nu: public GNAObject,
                  public TransformationBind<OscProb2nu> {
public:
  OscProb2nu();

  template <typename DerivedA, typename DerivedB>
  void probability(const Eigen::ArrayBase<DerivedA> &Enu,
                   Eigen::ArrayBase<DerivedB> &ret);
protected:
  variable<double> m_L;
  std::unique_ptr<OscillationVariables> m_param;
};
