#include <boost/format.hpp>

#include "WeightedSum.hh"
#include "TypesFunctions.hh"

WeightedSum::WeightedSum(const std::vector<std::string> &labels, const std::vector<std::string> &weight_labels) {
  if (labels.empty()) {
    return;
  }
  auto sum = transformation_("sum")
    .output("sum")
    .types(TypesFunctions::ifSame, TypesFunctions::pass<0>);

  if( labels.size()==weight_labels.size() ){
    sum.func(&WeightedSum::sumEq);
  }
  else if( labels.size()>weight_labels.size() ) {
    sum.func(&WeightedSum::sumArr);
  }
  else {
    sum.func(&WeightedSum::sumVal);
  }

  m_vars.resize(labels.size());
  for (size_t i = 0; i < labels.size(); ++i) {
    std::string wlabel;
    if ( weight_labels.empty() ) {
      wlabel = str(boost::format("weight_%1%")%labels[i]);
    }
    else{
      wlabel = weight_labels[i];
    }
    variable_(&m_vars[i], wlabel.data());
    sum.input(labels[i]);
  }
}

void WeightedSum::sumEq(Args args, Rets rets){
    rets[0].x = m_vars[0]*args[0].x;
    for (size_t i = 1; i < args.size(); ++i) {
      rets[0].x += m_vars[i]*args[i].x;
    }
}

void WeightedSum::sumArr(Args args, Rets rets){
    rets[0].x = m_vars[0]*args[0].x;
    size_t i = 1;
    for (; i < m_vars.size(); ++i) {
      rets[0].x += m_vars[i]*args[i].x;
    }
    for (; i < args.size(); ++i) {
      rets[0].x += args[i].x;
    }
}

void WeightedSum::sumVal(Args args, Rets rets){
    rets[0].x = m_vars[0]*args[0].x;
    size_t i = 1;
    for (; i < args.size(); ++i) {
      rets[0].x += m_vars[i]*args[i].x;
    }
    for (; i < m_vars.size(); ++i) {
      rets[0].x += m_vars[i].value();
    }
}

