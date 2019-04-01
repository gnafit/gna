#include "ViewRear.hh"
#include "TypeClasses.hh"

template<typename FloatType>
GNA::GNAObjectTemplates::ViewRearT<FloatType>::ViewRearT(size_t start, size_t len) : m_start(start), m_len(len) {
    init();
}

template<typename FloatType>
GNA::GNAObjectTemplates::ViewRearT<FloatType>::ViewRearT(typename GNAObjectT<FloatType,FloatType>::SingleOutput* output, size_t start, size_t len) :
ViewRearT(start, len)
{
    output->single() >> this->transformations.back().inputs.front();
}

template<typename FloatType>
void GNA::GNAObjectTemplates::ViewRearT<FloatType>::init(){
    this->transformation_("view")
         .input("original")
         .input("rear")
         .output("result")
         .types(new TypeClasses::CheckNdimT<FloatType>(1))
         .types(&ViewRearType::types)
         .func([](FunctionArgs& fargs){
                   static bool subsequent=false;
                   if(subsequent){
                       fargs.args[1];
                   }
                   else {
                       auto &args = fargs.args;
                       fargs.rets[0].x = args[0].x;
                       args[1];
                       subsequent=true;
                   }
               });

    //TODO: Add GPU option for preallocated buffer
}

template<typename FloatType>
void GNA::GNAObjectTemplates::ViewRearT<FloatType>::types(typename GNAObjectT<FloatType,FloatType>::TypesFunctionArgs& fargs){
    auto& args = fargs.args;

    /// Store the output datatype and check it has enough elements
    DataType dt_main = args[0];
    if(dt_main.kind==DataKind::Undefined){
        return;
    }

    auto required_length=m_start+m_len;
    if(dt_main.shape[0]<required_length)
    {
        throw args.error(dt_main, fmt::format("Transformation {0}: input {1} length should be at least {2}, got {3}",
                                              args.name(), 0, required_length, dt_main.shape[0]));
    }

    /// Allocate data if required
    if( !m_data || m_data->type.requiresReallocation(dt_main) ){
        m_data.reset(new Data<FloatType>(dt_main));
    }
    FloatType* buffer_main=m_data->buffer;

    auto& ret = (fargs.rets[0] = dt_main);
    ret.points().preallocated(buffer_main);

    m_datatype_sub=ret;
    m_datatype_sub.buffer=ret.buffer;

    /// Derive the proper datatype for inputs
    DataType dt_sub=dt_main;
    FloatType* buffer_sub=buffer_main+m_start;
    switch(dt_main.kind){
        case DataKind::Hist:
            dt_sub.hist().edges(m_len+1, dt_main.edges.data()+m_start).preallocated(buffer_sub);
            break;
        case DataKind::Points:
            dt_sub.points().shape(m_len).preallocated(buffer_sub);
            break;
        default:
            throw std::runtime_error("something is wrong (ViewRear)");
            break;
    }

    if(args[1].kind==DataKind::Undefined){
        return;
    }

    /// Check that input datatype is consistent
    if( args[1]!=dt_sub ){
        fprintf(stderr, "Main datatype:");
        dt_main.dump();

        fprintf(stderr, "Sub datatype: ");
        dt_sub.dump();

        fprintf(stderr, "Input datatype: ");
        args[1].dump();

        throw args.error(args[1], fmt::format("Transformation {0}: input {1} type is inconsistent with expectations", args.name(), 1));
    }

    /// Check that input data is not preallocatad
    auto& data_in=const_cast<Data<FloatType>&>(args.data(1));
    if(!data_in.allocated && data_in.buffer!=buffer_sub){
        throw args.error(args[1], fmt::format("Transformation {0}: input {1} may not be preallocated", args.name(), 1));
    }

    /// Reallocate input data
    /// FIXME: this is done in a very ugly way. The proper mechanism should be provided via framework.
    if(data_in.allocated){
        /// Patch the underlying datatype
        DataType& dt_in = const_cast<DataType&>(data_in.type);
        dt_in.buffer=static_cast<void*>(buffer_sub);

        /// Update the buffers
        data_in.allocated.reset(nullptr);
        data_in.init(buffer_sub);
    }
}

template class GNA::GNAObjectTemplates::ViewRearT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::GNAObjectTemplates::ViewRearT<float>;
#endif
