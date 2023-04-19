#pragma once

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

#include "config_vars.h"

class Rebin2: public GNAObject,
              public TransformationBind<Rebin2> {
public:
  Rebin2(size_t old_n, size_t new_n, double* old_edges, double* new_edges, int rounding);

  Eigen::SparseMatrix<double> getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd             getDenseMatrix() { return m_sparse_cache; }

private:
  void dump(size_t oldn, double* oldedges, size_t newn, double* newedges) const;
  void calcMatrix(TypesFunctionArgs& fargs);
  void calcSmear(FunctionArgs& fargs);

  double round(double num);

  std::vector<double> m_old_edges;
  std::vector<double> m_new_edges;
  double m_round_scale;

  bool m_initialized{false};
  Eigen::SparseMatrix<double> m_sparse_cache;
};
