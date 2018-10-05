#include "ArraySum.hh"
#include <algorithm>
#include <functional>


ArraySum::ArraySum(SingleOutput& out) {
    auto& taint = out.single().expose_taintflag();
    std::vector<changeable> deps;
    deps.push_back(std::ref(taint));
    m_accumulated = evaluable_<double>("out",
            [&out, this]() -> decltype(auto) {
                size_t offset = out.datatype().size();
                return std::accumulate(out.single().data(), out.single().data() + offset, 0.);}, 
                m_deps);
}

