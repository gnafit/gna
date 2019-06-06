#pragma once

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class Rebin: public GNAObject,
             public TransformationBind<Rebin> {
public:
  Rebin(size_t n, double* edges, int rounding);

  Eigen::SparseMatrix<double> getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd             getDenseMatrix() { return m_sparse_cache; }

private:
  void dump(size_t oldn, double* oldedges, size_t newn, double* newedges) const;
  void calcMatrix(const DataType& type);
  void calcSmear(FunctionArgs& fargs);
  double round(double num);

  std::vector<double> m_new_edges;
  double m_round_scale;

  bool m_initialized{false};
  Eigen::SparseMatrix<double> m_sparse_cache;
};
