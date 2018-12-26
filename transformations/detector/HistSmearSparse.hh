#pragma once

#include <vector>

#include "GNAObject.hh"
#include "GNAObjectBind1N.hh"
#include "Eigen/Sparse"

class HistSmearSparse: public GNAObjectBind1N,
                       public TransformationBind<HistSmearSparse> {
public:
  HistSmearSparse(bool propagate_matrix=false);

  Eigen::SparseMatrix<double> getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd             getDenseMatrix() { return m_sparse_cache; }

  TransformationDescriptor add_transformation(const std::string& name="");

protected:
  Eigen::SparseMatrix<double> m_sparse_cache;
  bool m_propagate_matrix{false};

private:
  void calcSmear(FunctionArgs& fargs);
};
