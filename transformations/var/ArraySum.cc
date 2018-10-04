#include "ArraySum.hh"
#include <algorithm>
#include <functional>



ArraySum::ArraySum(std::vector<std::string> names, std::string output_name):
    m_names(std::move(names)), m_output_name(std::move(output_name)) {

    m_vars.resize(m_names.size());
    for (size_t i=0; i < m_vars.size(); ++i) {
        variable_(&m_vars[i], m_names[i].c_str());
        m_deps.push_back(m_vars[i]);
    }

    auto trans = transformation_("arrsum")
        .input("arr")
        .output("accumulated")
        .types([](ArraySum* obj, TypesFunctionArgs& fargs){
                fargs.rets[0] = DataType().points().shape(1);})
        .func([](ArraySum* obj, FunctionArgs fargs) {
                double result = fargs.args[0].arr.sum();
                std::cout << result << std::endl;
                fargs.rets[0].arr(0) = result;
               });
}

void ArraySum::exposeEvaluable() {
    /* auto a = this->transformations.at(0).data()[0]; */
    m_accumulated = evaluable_(m_output_name.c_str(), [this]() -> decltype(this->transformations.at(0).data()[0]) {return this->transformations.at(0).data()[0];}, m_deps);
}
