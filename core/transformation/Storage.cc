#include "Storage.hh"

template class TransformationTypes::StorageT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::StorageT<float>;
#endif

