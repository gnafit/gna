#pragma once

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class HistSmearSparse: public GNAObject,
                       public TransformationBind<HistSmearSparse> {
public:
  HistSmearSparse(bool single=true);

  Eigen::SparseMatrix<double> getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd             getDenseMatrix() { return m_sparse_cache; }

protected:
  TransformationDescriptor add_smear();
  TransformationDescriptor add_smear(SingleOutput& matrix);

private:
  void calcSmear(FunctionArgs& fargs);

  Eigen::SparseMatrix<double> m_sparse_cache;

  bool m_single; /// restricts HistSmearSparse to contain only one transformation
};
