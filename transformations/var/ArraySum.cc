#include "ArraySum.hh"
#include <algorithm>
#include <functional>

template<typename FloatType>
GNA::GNAObjectTemplates::ArraySumT<FloatType>::ArraySumT::ArraySumT() {
  this->transformation_("arrsum")
    .input("array")
    .output("sum")
    .label("arrsum")
    .types(&ArraySumType::check)
    .func(&ArraySumType::sum);
}

template<typename FloatType>
GNA::GNAObjectTemplates::ArraySumT<FloatType>::ArraySumT(SingleOutput& out) : ArraySumT() {
  out.single() >> this->transformations.back().inputs.back();
}

template<typename FloatType>
GNA::GNAObjectTemplates::ArraySumT<FloatType>::ArraySumT(const std::string& name, SingleOutput& out) : ArraySumT(out) {
  ArraySumType::initialize(name);
}

template<typename FloatType>
void GNA::GNAObjectTemplates::ArraySumT<FloatType>::initialize(const std::string& name) {
    m_output_name = name;

    auto handle = this->t_[0];
    std::vector<changeable> deps{handle.getTaintflag()};
    m_accumulated = this->template evaluable_<FloatType>(m_output_name, [handle]() -> decltype(auto) {
                                       return handle[0].x(0);
                                       }, deps);
}

template<typename FloatType>
void GNA::GNAObjectTemplates::ArraySumT<FloatType>::check(TypesFunctionArgs& fargs){
    fargs.rets[0] = DataType().points().shape(1);
}

template<typename FloatType>
void GNA::GNAObjectTemplates::ArraySumT<FloatType>::sum(FunctionArgs& fargs){
    fargs.rets[0].x(0) = fargs.args[0].x.sum();
}

template class GNA::GNAObjectTemplates::ArraySumT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::GNAObjectTemplates::ArraySumT<float>;
#endif
