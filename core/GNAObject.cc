#include <iostream>

#include "GNAObject.hh"

void GNAObject::dump() {
  std::cerr << "GNAObject " << (void*)this << ", ";
  std::cerr << "type: " << typeid(*this).name() << std::endl;

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
