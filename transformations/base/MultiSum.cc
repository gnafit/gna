#include "MultiSum.hh"
#include "TypeClasses.hh"

using GNA::GNAObjectTemplates::MultiSumT;
using namespace TypeClasses;

template<typename FunctionArgsType> void multisum(FunctionArgsType& fargs);

/**
 * @brief Constructor.
 */
template<typename FloatType>
MultiSumT<FloatType>::MultiSumT() :
BindClass("sum", "item", "sum")
{
    add_transformation();
}

/**
 * @brief Construct MultiSumT from vector of SingleOutput instances
 */
template<typename FloatType>
MultiSumT<FloatType>::MultiSumT(const typename MultiSumT<FloatType>::OutputDescriptors& outputs) : MultiSumT() {
    this->add_inputs(outputs);
}

template<typename FloatType>
typename MultiSumT<FloatType>::TransformationDescriptor MultiSumT<FloatType>::add_transformation(const std::string& name){
    this->transformation_(new_transformation_name(name))
        .types(new CheckSameTypesT<FloatType>({0,-1}), new PassTypeT<FloatType>(0,{0,-1}))
        .func(&multisum<FunctionArgs>);
    add_output();
    add_input();
    set_open_input();
    return transformations.back();
}

template<typename FunctionArgsType>
void multisum(FunctionArgsType& fargs){
    auto& args=fargs.args;
    auto& rets=fargs.rets;

    int iret=-1;
    using ArrayViewType = typename FunctionArgsType::RetsType::DataType::ArrayViewType;
    ArrayViewType* data=nullptr;
    for (size_t jarg = 0; jarg < args.size(); ++jarg) {
        auto jmap=fargs.getMapping(jarg);
        if(jmap-iret > 0){
            iret=jmap;
            data=&(rets[iret].x);
            *data = args[jarg].x;
        }
        else{
            *data += args[jarg].x;
        }
    }
}

template class GNA::GNAObjectTemplates::MultiSumT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::MultiSumT<float>;
#endif
