#ifndef HISTNONLINEARITY_H
#define HISTNONLINEARITY_H

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class HistNonlinearity: public GNAObject,
                        public Transformation<HistNonlinearity> {
public:
  HistNonlinearity( bool propagate_matrix=true );

  Eigen::SparseMatrix<double> getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd             getDenseMatrix() { return m_sparse_cache; }

  void set( SingleOutput& bin_edges, SingleOutput& bin_edges_modified, SingleOutput& ntrue );
  void set( SingleOutput& bin_edges, SingleOutput& bin_edges_modified );
  void set( SingleOutput& ntrue );

  void set_range( double min, double max ) { m_range_min=min; m_range_max=max; }
  double get_range_min() { return m_range_min; }
  double get_range_max() { return m_range_max; }
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

#endif // HISTNONLINEARITY_H
