#include "WeightedSum.hh"
#include "TypesFunctions.hh"

WeightedSum::WeightedSum(const std::vector<std::string> &labels, const std::vector<std::string> &weight_labels)
  : WeightedSum(false, labels, weight_labels){ }

WeightedSum::WeightedSum(double fillvalue, const std::vector<std::string> &labels, const std::vector<std::string> &weight_labels)
  : WeightedSum(true, labels, weight_labels) { m_fillvalue=fillvalue; }

WeightedSum::WeightedSum(bool use_fillvalue, const std::vector<std::string> &labels, const std::vector<std::string> &weight_labels) {
  if (labels.empty()) {
    return;
  }

  if( weight_labels.size()>0u ){
    m_common = std::min(labels.size(), weight_labels.size());
  }
  else{
    m_common = labels.size();
  }

  auto sum = transformation_("sum")
    .output("sum")
    .label("wsum")
    .types(TypesFunctions::ifSame, TypesFunctions::pass<0>);

  if( use_fillvalue ){
    sum.func(&WeightedSum::sumFill);
  }
  else{
    sum.func(&WeightedSum::sum);
  }

  m_vars.resize(weight_labels.size());
  for (size_t i = 0; i < m_vars.size(); ++i) {
    variable_(&m_vars[i], weight_labels[i].data());
  }
  for (auto& label : labels) {
    sum.input(label);
  }
}

void WeightedSum::sum(Args args, Rets rets){
    rets[0].x = m_vars[0]*args[0].x;
    size_t i = 1;
    for (; i < m_common; ++i) {
      rets[0].x += m_vars[i]*args[i].x;
    }
    for (; i < args.size(); ++i) {
      rets[0].x += args[i].x;
    }
    for (; i < m_vars.size(); ++i) {
      rets[0].x += m_vars[i].value();
    }
}

void WeightedSum::sumFill(Args args, Rets rets){
    rets[0].x = m_fillvalue;
    size_t i = 0;
    for (; i < m_common; ++i) {
      rets[0].x += m_vars[i]*args[i].x;
    }
    for (; i < args.size(); ++i) {
      rets[0].x += args[i].x;
    }
    for (; i < m_vars.size(); ++i) {
      rets[0].x += m_vars[i].value();
    }
}
