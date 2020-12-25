#include "Histogram.hh"

Histogram::Histogram(size_t nbins, const double *edges, const double *data, bool fcn_copy)
  {
    m_edges = std::vector<double>{edges, edges+nbins+1};
    m_data = Eigen::Map<const Eigen::ArrayXd>(data, nbins);
    if( fcn_copy ){
      init_copy();
    }
    else {
      init();
    }
  }
