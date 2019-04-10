#include "ViewHistBased.hh"
#include "TypeClasses.hh"
#include "fmt/format.h"

#include <algorithm>
#include <iterator>

template<typename FloatType>
GNA::GNAObjectTemplates::ViewHistBasedT<FloatType>::ViewHistBasedT(FloatType threshold, FloatType ceiling) : m_threshold(threshold), m_ceiling(ceiling) {
    init();
}

template<typename FloatType>
GNA::GNAObjectTemplates::ViewHistBasedT<FloatType>::ViewHistBasedT(typename GNAObjectT<FloatType,FloatType>::SingleOutput& output, FloatType threshold, FloatType ceiling) :
ViewHistBasedT(threshold, ceiling)
{
    set(output);
}

template<typename FloatType>
void GNA::GNAObjectTemplates::ViewHistBasedT<FloatType>::init(){
    this->transformation_("hist")
         .input("hist", /*inactive=*/true)
         .output("hist_truncated")
         .types(new TypeClasses::CheckKindT<FloatType>(DataKind::Hist), new TypeClasses::CheckNdimT<FloatType>(1))
         .types(&ViewHistBasedType::histTypes)
         .func([](FunctionArgs& fargs){});

    add_transformation();
}

template<typename FloatType>
typename GNA::GNAObjectTemplates::ViewHistBasedT<FloatType>::TransformationDescriptor GNA::GNAObjectTemplates::ViewHistBasedT<FloatType>::add_transformation(const std::string& name){
    int n = this->transformations.size()-1;
    auto tname = name;
    if(!tname.size()){
        tname = n ? fmt::format("view_{:02d}", n) : "view";
    }
    this->transformation_(tname)
         .input("data")
         .output("view")
         .types(new TypeClasses::CheckNdimT<FloatType>(1))
         .types(&ViewHistBasedType::types)
         .func([](FunctionArgs& fargs){fargs.args.touch();});

    return this->transformations.back();
}

template<typename FloatType>
void GNA::GNAObjectTemplates::ViewHistBasedT<FloatType>::histTypes(typename GNAObjectT<FloatType,FloatType>::TypesFunctionArgs& fargs){
    auto& edges = fargs.args[0].edges;

    if( m_threshold && m_threshold.value()>edges.front() ){
        auto it_start = std::upper_bound(edges.begin(), edges.end(), m_threshold.value());
        if(it_start==edges.end()){
            throw fargs.args.error(fargs.args[0], "Threshold is too big");
        }
        m_start = std::distance(edges.begin(), it_start-1);
    }
    else{
        m_start = 0;
    }

    if( m_ceiling && m_ceiling.value()<edges.back() ){
        auto it_end = std::upper_bound(edges.begin(), edges.end(), m_ceiling.value());
        if(it_end==edges.end()){
            throw fargs.args.error(fargs.args[0], "Ceiling is too low");
        }
        m_len = std::distance(edges.begin(), it_end)-m_start.value();
    }
    else{
        m_len = edges.size()-1-m_start.value();
    }

    fargs.rets[0].hist().edges(m_len.value()+1, edges.data()+m_start.value());
}

template<typename FloatType>
void GNA::GNAObjectTemplates::ViewHistBasedT<FloatType>::types(typename GNAObjectT<FloatType,FloatType>::TypesFunctionArgs& fargs){
    if(!m_start || !m_len){
        return;
    }

    auto& args = fargs.args;
    auto& rets = fargs.rets;
    for (size_t i = 0; i < args.size(); ++i) {
        auto& arg = args[i];
        FloatType* buf = const_cast<FloatType*>(args.data(i).buffer)+m_start.value();
        auto required_length=m_start.value()+m_len.value();
        switch(arg.kind){
            case DataKind::Hist:
                rets[i].hist().edges(m_len.value()+1, arg.edges.data()+m_start.value()).preallocated(buf);
                break;
            case DataKind::Points:
                rets[i].points().shape(m_len.value()).preallocated(buf);
                break;
            default:
                continue;
                break;
        }
        if(arg.shape[0]<required_length)
        {
            throw fargs.args.error(arg, fmt::format("Transformation {0}: input {1} length should be at least {2}, got {3}",
                                                  args.name(), i, required_length, arg.shape[0]));
        }
    }
}

template class GNA::GNAObjectTemplates::ViewHistBasedT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::GNAObjectTemplates::ViewHistBasedT<float>;
#endif
