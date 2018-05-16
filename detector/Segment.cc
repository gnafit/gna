#include <boost/math/constants/constants.hpp>
#include "Segment.hh"


Segment::Segment() {
  variable_(&m_a, "Eres_a");
  variable_(&m_b, "Eres_b");
  variable_(&m_c, "Eres_c");
  transformation_(this, "normlayer")
      .output("normlayer")
      .types(Atypes::pass<0>,
         [](Segment *obj, Atypes args, Rtypes /*rets*/) {
           obj->m_datatype = args[0];
           obj->fillCache();
         })
       .func(&Segment::calweights);
}


std::vector<std::double> Segment::calweights() const noexcept {

    std::vector<std::double> weights;
    weights.clear();
    weights.push_back((m_a/17.2)*(m_a/17.2)*(m_a/17.2));
    weights.push_back((m_b/17.2)*(m_b/17.2)*(m_b/17.2)-(m_a/17.2)*(m_a/17.2)*(m_a/17.2));
    weights.push_back((m_c/17.2)*(m_c/17.2)*(m_c/17.2)-(m_b/17.2)*(m_b/17.2)*(m_b/17.2));
    
  return weights;
}



