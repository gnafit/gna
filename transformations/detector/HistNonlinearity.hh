#pragma once

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class HistNonlinearity: public GNAObject,
                        public TransformationBind<HistNonlinearity> {
public:
  HistNonlinearity(bool single=true, bool propagate_matrix=false);

  Eigen::SparseMatrix<double> getMatrix()      { return m_sparse_cache; }
  Eigen::MatrixXd             getDenseMatrix() { return m_sparse_cache; }

  void set( SingleOutput& bin_edges, SingleOutput& bin_edges_modified, SingleOutput& ntrue );
  void set( SingleOutput& bin_edges, SingleOutput& bin_edges_modified );
  void set( SingleOutput& ntrue );
  void set();

  TransformationDescriptor add();

  void set_range( double min, double max ) { m_range_min=min; m_range_max=max; }
  double get_range_min() { return m_range_min; }
  double get_range_max() { return m_range_max; }
private:
  void calcSmear(FunctionArgs& fargs);
  void calcMatrix(FunctionArgs& fargs);

  DataType m_datatype;

  Eigen::SparseMatrix<double> m_sparse_cache;

  bool m_initialized{false};
  bool m_propagate_matrix{false};

  double m_range_min{-1.e+100};
  double m_range_max{+1.e+100};

  bool m_single; /// restricts HistNonlinearity to contain only one transformation
};
