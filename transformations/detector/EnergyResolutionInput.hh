#pragma once

#include <vector>

#include "GNAObject.hh"
#include "HistSmearSparse.hh"
#include "Eigen/Sparse"

class EnergyResolutionInput: public HistSmearSparse,
                             public TransformationBind<EnergyResolutionInput> {
public:
  using TransformationBind<EnergyResolutionInput>::transformation_;
  EnergyResolutionInput(bool propagate_matrix=false);

  double resolution(double Etrue, double Erec, double RelSigma) const noexcept;

private:
  void calcMatrix(FunctionArgs& fargs);
};
