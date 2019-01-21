#pragma once

#include <vector>

#include "GNAObject.hh"
#include "HistSmearSparse.hh"
#include "Eigen/Sparse"

class EnergyResolution: public HistSmearSparse,
                        public TransformationBind<EnergyResolution> {
public:
  using TransformationBind<EnergyResolution>::transformation_;
  EnergyResolution(const std::vector<std::string>& pars, bool propagate_matrix=false);
  EnergyResolution(bool propagate_matrix=false);

  double relativeSigma(double Etrue) const noexcept;
  double resolution(double Etrue, double Erec) const noexcept;

private:
  void calcMatrix(FunctionArgs& fargs);
  void calcSmear(FunctionArgs& fargs);
  void getEdges(TypesFunctionArgs& fargs);

  variable<double> m_a, m_b, m_c;

  double const* m_edges=nullptr;
};
