#pragma once

#include <vector>

#include "GNAObject.hh"
#include <Eigen/Sparse>

class Npesmear: public GNAObject,
                       public TransformationBind<Npesmear> {
public:
  Npesmear( bool single=true );

  double resolution(double Etrue, double Erec) const noexcept;

  Eigen::SparseMatrix<double> getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd             getDenseMatrix() { return m_sparse_cache; }

  void add();
  void add(SingleOutput& hist);
private:
  void fillCache();
  void calcSmear(FunctionArgs& fargs);


  DataType m_datatype;

  size_t m_size;
  Eigen::SparseMatrix<double> m_sparse_cache;

  bool m_single; /// restricts Npesmear to contain only one transformation
};
