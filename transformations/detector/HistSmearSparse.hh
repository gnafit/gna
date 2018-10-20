#pragma once

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class HistSmearSparse: public GNAObject,
                       public TransformationBind<HistSmearSparse> {
public:
  HistSmearSparse(bool single=true, bool propagate_matrix=false);

  Eigen::SparseMatrix<double> getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd             getDenseMatrix() { return m_sparse_cache; }

  TransformationDescriptor add(SingleOutput& matrix);
  TransformationDescriptor add(bool connect_matrix);
  TransformationDescriptor add();

protected:
  Eigen::SparseMatrix<double> m_sparse_cache;
  bool m_propagate_matrix{false};

private:
  void calcSmear(FunctionArgs& fargs);

  bool m_single; /// restricts HistSmearSparse to contain only one transformation
};
