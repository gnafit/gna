#pragma once

#include <vector>

#include "HistSmearSparse.hh"
#include "Eigen/Sparse"

/**
 * @brief A class to smear the 1d histogram due to rescaling of the X axis
 *
 * HistNonlinearityC projects the shifted bins on the original binning.
 * The output histogram contains more bins: a distinct bin for each overlap
 * of smeared binning.
 * The output histogram is shared as two arrays: data and edges as the edges
 * are defined during the runtime.
 */
class HistNonlinearityC: public HistSmearSparse,
                         public TransformationBind<HistNonlinearityC> {
public:
  using TransformationBind<HistNonlinearityC>::transformation_;

  HistNonlinearityC(float nbins_factor, size_t nbins_extra, GNA::DataPropagation propagate_matrix=GNA::DataPropagation::Ignore);
  HistNonlinearityC(GNA::DataPropagation propagate_matrix=GNA::DataPropagation::Ignore) : HistNonlinearityC(2.2, 1, propagate_matrix) {};

  void set(SingleOutput& orig, SingleOutput& orig_proj, SingleOutput& mod_proj);

  void set_range( double min, double max ) { m_range_min=min; m_range_max=max; }
  double get_range_min() { return m_range_min; }
  double get_range_max() { return m_range_max; }

private:
  void calcSmear(FunctionArgs& fargs);
  void calcMatrix(FunctionArgs& fargs);
  void types(TypesFunctionArgs& fargs);

  bool m_initialized{false};

  double m_range_min{-1.e+100};
  double m_range_max{+1.e+100};

  double const* m_edges=nullptr;

  const float m_nbins_factor = 2.2;
  const size_t m_nbins_extra = 1;
  const double m_overflow_step = 10.0;

  // Eigen::SparseMatrix<double> m_sparse_rebin;
};
