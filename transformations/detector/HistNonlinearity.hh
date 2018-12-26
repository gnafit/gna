#pragma once

#include <vector>

#include "HistSmearSparse.hh"
#include "Eigen/Sparse"

class HistNonlinearity: public HistSmearSparse,
                        public TransformationBind<HistNonlinearity> {
public:
  using TransformationBind<HistNonlinearity>::transformation_;

  HistNonlinearity(bool propagate_matrix=false);

  void set( SingleOutput& bin_edges, SingleOutput& bin_edges_modified );

  void set_range( double min, double max ) { m_range_min=min; m_range_max=max; }
  double get_range_min() { return m_range_min; }
  double get_range_max() { return m_range_max; }
private:
  void calcSmear(FunctionArgs& fargs);
  void calcMatrix(FunctionArgs& fargs);

  bool m_initialized{false};

  double m_range_min{-1.e+100};
  double m_range_max{+1.e+100};
};
