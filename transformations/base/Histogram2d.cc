#include "Histogram2d.hh"
Histogram2d::Histogram2d(size_t xbins, const double *xedges, size_t ybins, const double *yedges, const double *data) :
    m_xedges{xedges, xedges+xbins+1},
    m_yedges{yedges, yedges+ybins+1},
    m_data(Eigen::Map<const Eigen::ArrayXXd>(data, xbins, ybins))
  {
    init();
  }
