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

    if (__builtin_expect(m_idx<0, 0)) {
        findStartingPoint(input, input + size);
    }

    std::fill(output, output+m_idx, 0.);
    std::partial_sum(input+m_idx, input+size, output+m_idx);
}

void PartialSum::findIdx(TypesFunctionArgs targs) {
    if (targs.args[0].kind == DataKind::Points) {
        this->m_idx = -1;
        return;
    }
    auto& edges = targs.args[0].edges;
    findStartingPoint(std::begin(edges), std::end(edges));
}

template <typename InputIterator>  
void PartialSum::findStartingPoint(InputIterator start, InputIterator end) {
    auto above_starting = std::lower_bound(start, end, m_starting_value);
    if (above_starting == end) {
        throw std::runtime_error("Starting value is larger then any bin in container");
    }
        this->m_idx = std::distance(start, above_starting);
}
