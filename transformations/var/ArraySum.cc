#include "ArraySum.hh"
#include <algorithm>
#include <functional>


SumToEvaluable::SumToEvaluable(const std::vector<double>& arr, std::string name): m_arr(arr), m_name(std::move(name)) {
    std::vector<changeable> deps;
    m_accumulated = evaluable_<double>(m_name.c_str(), [this]() {
            return std::accumulate(m_arr.begin(), m_arr.end(), 0.);
            }, deps);
}

ArraySum::ArraySum() {
    transformation_("sum")
        .input("arr")
        .output("accumulated")
        .types([](ArraySum* obj, TypesFunctionArgs fargs){
                fargs.rets[0] = DataType().points().shape(1);})
        .func([](ArraySum* obj, FunctionArgs fargs) {
                auto result = fargs.args[0].arr.sum();
                fargs.rets[0].arr = result;
               });
}

