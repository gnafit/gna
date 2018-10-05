#include "ArraySum.hh"
#include <algorithm>
#include <functional>


ArraySum::ArraySum(SingleOutput& out) {
    auto& taint = out.single().expose_taintflag();

    auto trans = transformation_("arrsum")
        .input("arr")
        .output("accumulated")
        .types([](ArraySum* obj, TypesFunctionArgs& fargs){
                fargs.rets[0] = DataType().points().shape(1);})
        .func([](ArraySum* obj, FunctionArgs fargs) {
                fargs.rets[0].arr(0) = fargs.args[0].arr.sum();
               });
    t_[0].inputs()[0].connect(out.single());

    std::vector<changeable> deps;
    deps.push_back(taint);
    m_accumulated = evaluable_<double>("out",
            [&out]() -> decltype(auto) {
                size_t offset = out.datatype().size();
                return std::accumulate(out.single().data(), out.single().data() + offset, 0.);},
                m_deps);
    m_accumulated.subscribe(taint);
    printf("depends %i\n", int(m_accumulated.depends(taint)));
}

