#include "Points.hh"

template class GNA::GNAObjectTemplates::PointsT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::PointsT<float>;
#endif
