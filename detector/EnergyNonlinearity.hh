#ifndef ENERGYNONLINEARITY_H
#define ENERGYNONLINEARITY_H

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class EnergyNonlinearity: public GNAObject,
                          public Transformation<EnergyNonlinearity> {
public:
  EnergyNonlinearity( bool propagate_matrix=false );

  Eigen::SparseMatrix<double>& getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd              getDenseMatrix() { return m_sparse_cache; }

  void set( SingleOutput& bin_edges, SingleOutput& bin_edges_modified, SingleOutput& ntrue );

  void set_range( double min, double max ) { m_range_min=min; m_range_max=max; }
private:
  void calcSmear(Args args, Rets rets);
  void calcMatrix(Args args, Rets rets);

  DataType m_datatype;

  size_t m_size;
  Eigen::SparseMatrix<double> m_sparse_cache;

  bool m_initialized{false};
  bool m_propagate_matrix{false};

  double m_range_min{-1.e+100};
  double m_range_max{+1.e+100};
};

#endif // ENERGYNONLINEARITY_H
