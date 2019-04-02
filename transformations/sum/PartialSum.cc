#include "PartialSum.hh"
#include "TypeClasses.hh"
#include <numeric>
#include <fmt/ostream.h>

PartialSum::PartialSum() {
    init();
}

PartialSum::PartialSum(double initial_value):
m_initial_value(initial_value)
{
    init();
}

PartialSum::PartialSum(double initial_value, double threshold):
m_threshold(threshold),
m_initial_value(initial_value)
{
    init();
}

PartialSum::PartialSum(double threshold, bool):
m_threshold(threshold)
{
    init();
}

void PartialSum::init() {
    transformation_("reduction")
        .input("inp")
        .output("out")
        .types(new TypeClasses::CheckNdimT<double>(1))
        .types( &PartialSum::make_Points,
                &PartialSum::findIdx
            )
        .func(&PartialSum::calc);
}

void PartialSum::make_Points(TypesFunctionArgs targs) {
    size_t size=targs.args[0].size();
    if(m_initial_value){
        ++size;
    }
    targs.rets[0] = DataType().points().shape(size);
}

void PartialSum::calc(FunctionArgs fargs) {
    auto& inputs = fargs.args[0];
    auto* input = inputs.buffer;
    auto  size = inputs.x.size();
    auto& ret=fargs.rets[0];
    auto* output = ret.buffer;

    if(m_initial_value)
    {
        std::fill(output, output+m_idx+1, m_initial_value.value());
        std::partial_sum(input+m_idx, input+size, output+m_idx+1);
        ret.x.tail(size-m_idx)+=m_initial_value.value();
    }
    else{
        std::fill(output, output+m_idx, 0.0);
        std::partial_sum(input+m_idx, input+size, output+m_idx);
    }
}

void PartialSum::findIdx(TypesFunctionArgs targs) {
    if (targs.args[0].kind == DataKind::Points) {
        this->m_idx = 0;
        return;
    }
    const auto& edges = targs.args[0].edges;
    if(this->m_threshold){
        findStartingPoint(std::begin(edges), std::end(edges));
    }
}

template <typename InputIterator>
void PartialSum::findStartingPoint(InputIterator start, InputIterator end) {
    auto above_starting = std::lower_bound(start, end, m_threshold.value());
    if (above_starting == end) {
        throw std::runtime_error("Starting value is larger then any bin in container");
    }
    this->m_idx = std::distance(start, above_starting);
}
