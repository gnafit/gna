#include "PartialSum.hh"
#include <numeric>
#include <fmt/ostream.h>

PartialSum::PartialSum(double starting_value): m_starting_value{starting_value} {
    transformation_("reduction")
        .input("inp")
        .output("out")
        .types(TypesFunctions::passAll,
                &PartialSum::findIdx
            )
        .func(&PartialSum::calc);
}

void PartialSum::calc(FunctionArgs fargs) {
    auto& inputs = fargs.args[0];
    auto* input = inputs.buffer;
    auto  size = inputs.x.size();
    auto* output = fargs.rets[0].buffer;

    std::fill(output, output+m_idx, 0.);
    std::partial_sum(input+m_idx, input+size, output+m_idx);
}

void PartialSum::findIdx(TypesFunctionArgs targs) {
    auto& edges = targs.args[0].edges;
    auto above_starting = std::lower_bound(std::begin(edges), std::end(edges), m_starting_value);
    if (above_starting == edges.end()) {
        throw std::runtime_error("Starting value is larger then any bin in histogram");
    }
    this->m_idx = std::distance(std::begin(edges), above_starting);
}
