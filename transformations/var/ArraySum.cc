#include "ArraySum.hh"
#include <algorithm>
#include <functional>


ArraySum::ArraySum(const std::string& name, SingleOutput& out) {
    ArraySum::initialize(name, out);
}

void ArraySum::initialize(const std::string& name, SingleOutput& out) {
    m_output_name = name;

    std::vector<changeable> deps{out.single().expose_taintflag()};
    m_accumulated = evaluable_<double>(m_output_name,
            [&out]() -> decltype(auto) {
                size_t offset = out.datatype().size();
                auto* data = out.single().data();
                return std::accumulate(data, data + offset, 0.);},
                deps);
}
