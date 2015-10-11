#include <boost/format.hpp>

#include "WeightedSum.hh"

WeightedSum::WeightedSum(const std::vector<std::string> &labels) {
  if (labels.empty()) {
    return;
  }
  auto sum = transformation_(this, "sum")
    .output("sum", DataType().points().any())
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
    variable_(&m_vars[i], (boost::format("weight_%1%") % labels[i]).str());
    sum.input(labels[i], DataType().points().any());
  }
}

