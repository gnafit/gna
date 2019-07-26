#include "ViewHistBased.hh"
#include "TypeClasses.hh"
#include "fmt/format.h"

#include <algorithm>
#include <iterator>

template<typename FloatType>
GNA::GNAObjectTemplates::ViewHistBasedT<FloatType>::ViewHistBasedT(FloatType threshold, FloatType ceiling) :
GNAObjectBind1N<FloatType>("view", "data_in", "view", 1, 0, 0),
m_threshold(threshold), m_ceiling(ceiling)
{
    init();

    this->add_transformation();
    this->add_input();
    this->set_open_input();
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
}

template<typename FloatType>
typename GNA::GNAObjectTemplates::ViewHistBasedT<FloatType>::TransformationDescriptor GNA::GNAObjectTemplates::ViewHistBasedT<FloatType>::add_transformation(const std::string& name){
    this->transformation_(this->new_transformation_name(name))
         .types(new TypeClasses::CheckNdimT<FloatType>(1))
         .types(&ViewHistBasedType::types)
         .func([](FunctionArgs& fargs){fargs.args.touch();});

    this->reset_open_input();

    return this->transformations.back();
}

template<typename FloatType>
void GNA::GNAObjectTemplates::ViewHistBasedT<FloatType>::histTypes(typename GNAObjectT<FloatType,FloatType>::TypesFunctionArgs& fargs){
    auto& arg = fargs.args[0];
    auto& edges = arg.edges;
    m_full_length = arg.size();

    if( m_threshold && m_ceiling && m_ceiling.value()<m_threshold.value() ){
        throw fargs.args.error(fargs.args[0], "Threshold is above ceiling");
    }

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
        auto it_end = std::lower_bound(edges.begin(), edges.end(), m_ceiling.value());
        if(it_end==edges.end()){
            throw fargs.args.error(fargs.args[0], "Ceiling is too low");
        }
        m_len = std::distance(edges.begin(), it_end)-m_start.value();
    }
    else{
        m_len = edges.size()-1-m_start.value();
    }

    if(m_len && m_len.value()<1){
        throw fargs.args.error(fargs.args[0], "View should have at least one bin as base.");
    }

    fargs.rets[0].hist().edges(m_len.value()+1, edges.data()+m_start.value());

    for (size_t i = 1; i < this->transformations.size(); ++i) {
        this->transformations[i].updateTypes();
    }
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
        size_t argsize = arg.size();
        switch(arg.kind){
            case DataKind::Hist:
                if(argsize!=m_full_length){
                    throw fargs.args.error(arg, fmt::format("Transformation {0}: input {1} length should be {2}, got {3}",
                                                          args.name(), i, m_full_length, arg.shape[0]));
                }
                // Bin centers case
                rets[i].hist().edges(m_len.value()+1, arg.edges.data()+m_start.value()).preallocated(buf);
                break;
            case DataKind::Points:
                if(argsize==m_full_length){
                    // Bin centers case
                    rets[i].points().shape(m_len.value()).preallocated(buf);
                }
                else if(argsize==m_full_length+1){
                    // Bin edges case
                    rets[i].points().shape(m_len.value()+1).preallocated(buf);
                }
                else {
                    throw fargs.args.error(arg, fmt::format("Transformation {0}: input {1} length should be {2} or {3}, got {4}",
                                                          args.name(), i, m_full_length, m_full_length+1, arg.shape[0]));
                }
                break;
            default:
                continue;
                break;
        }
    }
}

template class GNA::GNAObjectTemplates::ViewHistBasedT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::GNAObjectTemplates::ViewHistBasedT<float>;
#endif
