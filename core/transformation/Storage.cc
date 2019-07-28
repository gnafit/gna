#include "Storage.hh"

template struct TransformationTypes::StorageT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template struct TransformationTypes::StorageT<float>;
#endif

