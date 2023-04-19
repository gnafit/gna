#pragma once

#include <vector>

#include "HistSmearSparse.hh"
#include "Eigen/Sparse"

/**
 * @brief A class to smear the 1d histogram due to rescaling of the X axis
 *
 * HistNonlinearityD projects the shifted bins on the binning, passed as input
 */
class HistNonlinearityD: public HistSmearSparse,
                         public TransformationBind<HistNonlinearityD> {
public:
  using TransformationBind<HistNonlinearityD>::transformation_;

  HistNonlinearityD(GNA::DataPropagation propagate_matrix=GNA::DataPropagation::Ignore);

  void set(SingleOutput& orig, SingleOutput& orig_proj, SingleOutput& mod, SingleOutput& mod_proj);

  void set_range( double min, double max ) { m_range_min=min; m_range_max=max; }
  double get_range_min() { return m_range_min; }
  double get_range_max() { return m_range_max; }
private:
  void calcSmear(FunctionArgs& fargs);
  void calcMatrix(FunctionArgs& fargs);
  void getEdges(TypesFunctionArgs& fargs);

  bool m_initialized{false};

  double m_range_min{-1.e+100};
  double m_range_max{+1.e+100};

  double const* m_edges_in=nullptr;
  double const* m_edges_out=nullptr;
};
