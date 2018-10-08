#include "ArraySum.hh"
#include <algorithm>
#include <functional>


ArraySum::ArraySum(const std::string& name, SingleOutput& out) {
    ArraySum::initialize(name, out);
}

void ArraySum::initialize(const std::string& name, SingleOutput& out) {
    m_output_name = name;
    //auto trans = transformation_("arrsum")
        //.input("arr")
        //.output("accumulated")
        //.types([](ArraySum* obj, TypesFunctionArgs& fargs){
                //fargs.rets[0] = DataType().points().shape(1);})
        //.func([](ArraySum* obj, FunctionArgs fargs) {
                //fargs.rets[0].arr(0) = fargs.args[0].arr.sum();
               //});
    //t_[0].inputs()[0].connect(out.single());

    std::vector<changeable> deps{out.single().expose_taintflag()};
    m_accumulated = evaluable_<double>(m_output_name,
            [&out]() -> decltype(auto) {
                size_t offset = out.datatype().size();
                auto* data = out.single().data();
                return std::accumulate(data, data + offset, 0.);},
                deps);
}
