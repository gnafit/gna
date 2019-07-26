#include "Snapshot.hh"
#include "TypeClasses.hh"

using FunctionArgs = TransformationTypes::FunctionArgsT<double,double>;

template<typename FloatType>
GNA::GNAObjectTemplates::SnapshotT<FloatType>::SnapshotT() :
GNAObjectBind1N<FloatType>("snapshot", "source", "result", 0, 0, 0)
{
    this->add_transformation();
    this->add_input();
    this->set_open_input();
}

template<typename FloatType>
GNA::GNAObjectTemplates::SnapshotT<FloatType>::SnapshotT(typename GNA::GNAObjectTemplates::SnapshotT<FloatType>::SingleOutput& output) :
GNAObjectBind1N<FloatType>("snapshot", "source", "result", 0, 0, 0)
{
    this->add_transformation();
    this->add_input(output);
}

template<typename FloatType>
typename GNA::GNAObjectTemplates::SnapshotT<FloatType>::TransformationDescriptor GNA::GNAObjectTemplates::SnapshotT<FloatType>::add_transformation(const std::string& name){
    this->transformation_("snapshot")
        .types(new TypeClasses::PassEachTypeT<FloatType>())
        .func(&Snapshot::makeSnapshot)
        ;
    return this->transformations.back();
}

template<typename FloatType>
void GNA::GNAObjectTemplates::SnapshotT<FloatType>::makeSnapshot(typename GNA::GNAObjectTemplates::SnapshotT<FloatType>::FunctionArgs& fargs){
    auto& args = fargs.args;
    auto& rets = fargs.rets;
    for (size_t i = 0; i < args.size(); ++i) {
        rets[i].x = args[i].x;
    }
    rets.untaint();
    rets.freeze();
}

template class GNA::GNAObjectTemplates::SnapshotT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::GNAObjectTemplates::SnapshotT<float>;
#endif
