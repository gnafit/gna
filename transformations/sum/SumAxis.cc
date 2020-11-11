#include "SumAxis.hh"
#include "TypeClasses.hh"

void SumRows(TransformationTypes::FunctionArgsT<double,double> fargs){
    auto& args = fargs.args;
    auto& rets = fargs.rets;

    for (size_t i = 0; i < args.size(); ++i) {
        rets[i].arr = args[i].arr2d.colwise().sum();
    }
}

void SumColumns(TransformationTypes::FunctionArgsT<double,double> fargs){
    auto& args = fargs.args;
    auto& rets = fargs.rets;

    for (size_t i = 0; i < args.size(); ++i) {
        rets[i].arr = args[i].arr2d.rowwise().sum();
    }
}

SumAxis::SumAxis(size_t axis, bool uninitialized): GNAObjectBind1N<double>("sumaxis", "source", "result", 0, 0, 0), m_axis{axis}
{
    if(uninitialized) return;
    add_transformation();
    add_input();
    set_open_input();
}

SumAxis::SumAxis(size_t axis, SingleOutput& output): GNAObjectBind1N<double>("sumaxis", "source", "result", 0, 0, 0), m_axis{axis}
{
    add_transformation();
    add_input(output);
}

TransformationDescriptor SumAxis::add_transformation(const std::string& name){
    transformation_(new_transformation_name(name))
        .types(new TypeClasses::CheckNdimT<double>(2))
        .types(&SumAxis::SetTypes)
        .func(m_axis==0 ? &SumRows : &SumColumns);

    reset_open_input();
    return transformations.back();
}

void SumAxis::SetTypes(SumAxis::TypesFunctionArgs& fargs){
    auto& args = fargs.args;
    auto& rets = fargs.rets;
    size_t select = m_axis==0 ? 1u : 0u;
    for (size_t i = 0; i < args.size(); ++i) {
        auto& arg=args[i];
        if (!arg.defined()) continue;

        args[i].dump();
        if(arg.kind==DataKind::Hist){
            rets[i].hist().bins(arg.shape[select]).edges(arg.edgesNd[select]);
        }
        else{
            rets[i].points().shape(arg.shape[select]);
        }
    }
}

