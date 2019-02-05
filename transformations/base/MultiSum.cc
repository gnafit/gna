#include "MultiSum.hh"
#include "TypesFunctions.hh"

using FunctionArgs = TransformationTypes::FunctionArgsT<double,double>;
void multisum(FunctionArgs& fargs);

/**
 * @brief Constructor.
 */
MultiSum::MultiSum() :
GNAObjectBindMN("sum", "item", "sum")
{
    add_transformation();
}

/**
 * @brief Construct MultiSum from vector of SingleOutput instances
 */
MultiSum::MultiSum(const OutputDescriptor::OutputDescriptors& outputs) : MultiSum() {
    add_inputs(outputs);
}

TransformationDescriptor MultiSum::add_transformation(const std::string& name){
    transformation_(new_transformation_name(name))
        .types(TypesFunctions::ifSame, TypesFunctions::passToRange<0,0,-1>)
        .func(multisum);
    add_output();
    add_input();
    set_open_input();
    return transformations.back();
}

void multisum(FunctionArgs& fargs){
    auto& args=fargs.args;
    auto& rets=fargs.rets;

    int iret=-1;
    using ArrayViewType = typename FunctionArgs::RetsType::DataType::ArrayViewType;
    ArrayViewType* data=nullptr;
    for (size_t jarg = 0; jarg < args.size(); ++jarg) {
        auto jmap=fargs.getMapping(jarg);
        if(jmap-iret == 1){
            iret=jmap;
            data=&(rets[iret].x);
            *data = args[jarg].x;
        }
        else{
            *data += args[jarg].x;
        }
    }
}
