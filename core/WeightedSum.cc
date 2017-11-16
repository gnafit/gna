#include <boost/format.hpp>

#include "WeightedSum.hh"

WeightedSum::WeightedSum(const std::vector<std::string> &labels, const std::vector<std::string> &weight_labels) {
  if (labels.empty()) {
    return;
  }
  if( !weight_labels.empty() && labels.size()!=weight_labels.size() ){
      throw std::runtime_error( "Incompartible labels weight_labels lists" );
  }
  auto sum = transformation_(this, "sum")
    .output("sum")
    .types(Atypes::ifSame, Atypes::pass<0>)
    .func([] (WeightedSum *obj, Args args, Rets rets) {
        rets[0].x = obj->m_vars[0]*args[0].x;
        for (size_t i = 1; i < args.size(); ++i) {
          rets[0].x += obj->m_vars[i]*args[i].x;
        }
      })
  ;
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

