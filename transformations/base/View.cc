#include "View.hh"
#include "TypeClasses.hh"

template<typename FloatType>
GNA::GNAObjectTemplates::ViewT<FloatType>::ViewT(size_t start, size_t len) : m_start(start), m_len(len) {
    init();
}

template<typename FloatType>
GNA::GNAObjectTemplates::ViewT<FloatType>::ViewT(SingleOutput* output, size_t start, size_t len) :
ViewT(start, len)
{
    output->single() >> this->transformations.back().inputs.back();
}

template<typename FloatType>
void GNA::GNAObjectTemplates::ViewT<FloatType>::init(){
    this->transformation_("view")
         .input("data")
         .output("view")
         .types(new TypeClasses::CheckNdimT<FloatType>(1))
         .types(&ViewType::types)
         .func([](FunctionArgs& fargs){fargs.args.touch();});

    //TODO: Add GPU option for preallocated buffer
}

template<typename FloatType>
void GNA::GNAObjectTemplates::ViewT<FloatType>::types(TypesFunctionArgs& fargs){
    auto& args = fargs.args;
    auto& rets = fargs.rets;
    for (size_t i = 0; i < args.size(); ++i) {
        auto& arg = args[i];
        FloatType* buf = const_cast<FloatType*>(args.data(i).buffer)+m_start;
        auto required_length=m_start+m_len;
        switch(arg.kind){
            case DataKind::Hist:
                rets[i].hist().edges(m_len+1, arg.edges.data()+m_start).preallocated(buf);
                break;
            case DataKind::Points:
                rets[i].points().shape(m_len).preallocated(buf);
                break;
            default:
                continue;
                break;
        }
        if(arg.shape[0]<required_length)
        {
            throw args.error(arg, fmt::format("Transformation {0}: input {1} length should be at least {2}, got {3}",
                                                  args.name(), i, required_length, arg.shape[0]));
        }
    }
}

template class GNA::GNAObjectTemplates::ViewT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::GNAObjectTemplates::ViewT<float>;
#endif
