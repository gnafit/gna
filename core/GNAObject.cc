#include <iostream>

#include <boost/type_index.hpp>
#include "GNAObject.hh"

template<typename SourceFloatType, typename SinkFloatType>
void GNAObjectT<SourceFloatType,SinkFloatType>::dumpObj() {
  std::cerr << "GNAObject " << (void*)this << ", ";
  std::cerr << "type: " << boost::typeindex::type_id<decltype(*this)>().pretty_name() << std::endl;
  /* std::cerr << "type: " << typeid(*this).name() << std::endl; */
  /* boost::typeindex::type_id<T>().pretty_name() */

  std::cerr << "variables (" << variables.size() << "):" << std::endl;
  for (size_t i = 0; i < variables.size(); ++i) {
    std::cerr << "  " << i << ": ";
    variables.at(i).dump();
  }
  std::cerr << "evaluables (" << evaluables.size() << "):" << std::endl;
  for (size_t i = 0; i < evaluables.size(); ++i) {
    std::cerr << "  " << i << ": ";
    evaluables.at(i).dump();
  }
  std::cerr << "transformations (" << t_.size() << "):" << std::endl;
  for (size_t i = 0; i < t_.size(); ++i) {
    std::cerr << "  " << i << ": ";
    t_[i].dump();
  }
}

template class GNAObjectT<double,double>;
template class GNASingleObjectT<double,double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNAObjectT<float,float>;
  template class GNASingleObjectT<float,float>;
#endif
