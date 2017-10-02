#ifndef ENERGYNONLINEARITY_H
#define ENERGYNONLINEARITY_H

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class EnergyNonlinearity: public GNASingleObject,
                          public Transformation<EnergyNonlinearity> {
public:
  EnergyNonlinearity();

  Eigen::SparseMatrix<double>& getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd              getDenseMatrix() { return m_sparse_cache; }
private:
  //void fillCache();
  void calcSmear(Args args, Rets rets);

  //variable<double> m_a, m_b, m_c;

  DataType m_datatype;

  size_t m_size;
  Eigen::SparseMatrix<double> m_sparse_cache;
};

#endif // ENERGYNONLINEARITY_H
