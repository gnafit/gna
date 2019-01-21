#include "ArraySum.hh"
#include <algorithm>
#include <functional>


ArraySum::ArraySum(const std::string& name, SingleOutput& out) {
  transformation_("arrsum")
    .input("array")
    .output("sum")
    .label("arrsum")
    .types(&ArraySum::check)
    .func(&ArraySum::sum);

  out.single() >> transformations.back().inputs.back();

  ArraySum::initialize(name);
}

void ArraySum::initialize(const std::string& name) {
    m_output_name = name;

    auto handle = t_[0];
    std::vector<changeable> deps{handle.expose_taintflag()};
    m_accumulated = evaluable_<double>(m_output_name, [handle]() -> decltype(auto) {
                                       return handle[0].x(0);
                                       }, deps);
}

void ArraySum::check(TypesFunctionArgs& fargs){
    fargs.rets[0] = DataType().points().shape(1);
}

void ArraySum::sum(FunctionArgs& fargs){
    fargs.rets[0].x(0) = fargs.args[0].x.sum();
}
