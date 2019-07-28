#include "FillLike.hh"

template class GNA::GNAObjectTemplates::FillLikeT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::FillLikeT<float>;
#endif
