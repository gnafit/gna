#include "TypeClasses.hh"

template class TypeClasses::TypeClassT<double>;
template class TypeClasses::CheckSameTypesT<double>;
template class TypeClasses::PassTypeT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TypeClasses::TypeClassT<float>;
  template class TypeClasses::CheckSameTypesT<float>;
  template class TypeClasses::PassTypeT<float>;
#endif
