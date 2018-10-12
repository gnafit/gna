#pragma once

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class EnergyResolution: public GNAObject,
                        public TransformationBind<EnergyResolution> {
public:
  EnergyResolution(const std::vector<std::string>& pars, bool single=true, bool propagate_matrix=false);
  EnergyResolution(bool single=true, bool propagate_matrix=false);

  double relativeSigma(double Etrue) const noexcept;
  double resolution(double Etrue, double Erec) const noexcept;

  Eigen::SparseMatrix<double> getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd             getDenseMatrix() { return m_sparse_cache; }

  TransformationDescriptor add();
  TransformationDescriptor add(SingleOutput& hist);
private:
  void calcMatrix(FunctionArgs& fargs);
  void calcSmear(FunctionArgs& fargs);

  variable<double> m_a, m_b, m_c;

  Eigen::SparseMatrix<double> m_sparse_cache;
  bool m_propagate_matrix{false};

  bool m_single; /// restricts EnergyResolution to contain only one transformation
};
