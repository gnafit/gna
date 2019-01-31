#include "Points.hh"

template class PointsT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class PointsT<float>;
#endif
