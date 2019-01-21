#pragma once

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class EnergyResolutionC: public GNAObject,
                         public TransformationBind<EnergyResolutionC> {
public:
  EnergyResolutionC( bool single=true );

  double relativeSigma(double Etrue) const noexcept;
  double resolution(double Etrue, double Erec) const noexcept;

  Eigen::SparseMatrix<double> getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd             getDenseMatrix() { return m_sparse_cache; }

  void add();
  void add(SingleOutput& hist);
private:
  void fillCache();
  void calcSmear(FunctionArgs& fargs);

  variable<double> m_a, m_b, m_c;

  DataType m_datatype;

  size_t m_size;
  Eigen::SparseMatrix<double> m_sparse_cache;

  bool m_single; /// restricts EnergyResolutionC to contain only one transformation
};
