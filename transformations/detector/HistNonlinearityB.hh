#pragma once

#include <vector>

#include "HistSmearSparse.hh"
#include "Eigen/Sparse"

/**
 * @brief A class to smear the 1d histogram due to rescaling of the X axis
 *
 * HistNonlinearityB projects the shifted bins on the original binning
 */
class HistNonlinearityB: public HistSmearSparse,
                         public TransformationBind<HistNonlinearityB> {
public:
  using TransformationBind<HistNonlinearityB>::transformation_;

  HistNonlinearityB(GNA::DataPropagation propagate_matrix=GNA::DataPropagation::Ignore);

  void set(SingleOutput& orig, SingleOutput& orig_proj, SingleOutput& mod_proj);

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

  double const* m_edges=nullptr;
};
