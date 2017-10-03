#ifndef ENERGYNONLINEARITY_H
#define ENERGYNONLINEARITY_H

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class EnergyNonlinearity: public GNAObject,
                          public Transformation<EnergyNonlinearity> {
public:
  EnergyNonlinearity();

  Eigen::SparseMatrix<double>& getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd              getDenseMatrix() { return m_sparse_cache; }

  void set( SingleOutput& bin_edges, SingleOutput& bin_edges_modified, SingleOutput& ntrue );
private:
  void calcSmear(Args args, Rets rets);
  void calcMatrix(Args args, Rets rets);

  DataType m_datatype;

  size_t m_size;
  Eigen::SparseMatrix<double> m_sparse_cache;

  bool m_initialized{false};
};

#endif // ENERGYNONLINEARITY_H
