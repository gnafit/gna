#include "PartialSum.hh"
#include <numeric>

PartialSum::PartialSum() {
    transformation_("reduction")
        .input("inp")
        .output("out")
        .types(TypesFunctions::passAll)
        .func(&PartialSum::calc);
}

void PartialSum::calc(FunctionArgs fargs) {
    auto* input = fargs.args[0].buffer;
    auto* output = fargs.rets[0].buffer;
    auto size = fargs.args[0].x.size();
    std::partial_sum(input, input + size, output);
}
