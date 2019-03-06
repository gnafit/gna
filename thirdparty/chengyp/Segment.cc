#include "Segment.hh"
#include "TypesFunctions.hh"
#include <boost/math/constants/constants.hpp>


Segment::Segment() {
  variable_(&m_a, "Eres_a");
  variable_(&m_b, "Eres_b");
  variable_(&m_c, "Eres_c");
  this->transformation_("normlayer")
      .output("normlayer")
      .types(TypesFunctions::pass<0>)
      .types([](Segment *obj, TypesFunctionArgs& fargs) {
           obj->m_datatype = fargs.args[0];
           obj->fillCache();
         })
       .func(&Segment::calweights);
}


std::vector<std::double> Segment::calweights(FunctionArgs& args) const noexcept {
  std::vector<double> weights;
  weights.clear();
  weights.push_back((m_a/17.2)*(m_a/17.2)*(m_a/17.2));
  weights.push_back((m_b/17.2)*(m_b/17.2)*(m_b/17.2)-(m_a/17.2)*(m_a/17.2)*(m_a/17.2));
  weights.push_back((m_c/17.2)*(m_c/17.2)*(m_c/17.2)-(m_b/17.2)*(m_b/17.2)*(m_b/17.2));

  return weights;
}



