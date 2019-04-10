#include "ViewRear.hh"
#include "TypeClasses.hh"

template<typename FloatType>
GNA::GNAObjectTemplates::ViewRearT<FloatType>::ViewRearT(size_t start) : m_start(start) {
    init();
}

template<typename FloatType>
GNA::GNAObjectTemplates::ViewRearT<FloatType>::ViewRearT(typename GNAObjectT<FloatType,FloatType>::SingleOutput* output, size_t start) :
ViewRearT(start)
{
    output->single() >> this->transformations.back().inputs.front();
}

template<typename FloatType>
GNA::GNAObjectTemplates::ViewRearT<FloatType>::ViewRearT(size_t start, FloatType fill_value) : m_start(start), m_fill_value(fill_value) {
    init();
}

template<typename FloatType>
GNA::GNAObjectTemplates::ViewRearT<FloatType>::ViewRearT(typename GNAObjectT<FloatType,FloatType>::SingleOutput* output, size_t start, FloatType fill_value) :
ViewRearT(start, fill_value)
{
    output->single() >> this->transformations.back().inputs.front();
}

template<typename FloatType>
GNA::GNAObjectTemplates::ViewRearT<FloatType>::ViewRearT(size_t start, size_t len) :
m_start(start),
m_len(len)
{
    init();
}

template<typename FloatType>
GNA::GNAObjectTemplates::ViewRearT<FloatType>::ViewRearT(typename GNAObjectT<FloatType,FloatType>::SingleOutput* output, size_t start, size_t len) :
ViewRearT(start, len)
{
    output->single() >> this->transformations.back().inputs.front();
}

template<typename FloatType>
GNA::GNAObjectTemplates::ViewRearT<FloatType>::ViewRearT(size_t start, size_t len, FloatType fill_value) :
m_start(start),
m_len(len),
m_fill_value(fill_value)
{
    init();
}

template<typename FloatType>
GNA::GNAObjectTemplates::ViewRearT<FloatType>::ViewRearT(typename GNAObjectT<FloatType,FloatType>::SingleOutput* output, size_t start, size_t len, FloatType fill_value) :
ViewRearT(start, len, fill_value)
{
    output->single() >> this->transformations.back().inputs.front();
}

template<typename FloatType>
void GNA::GNAObjectTemplates::ViewRearT<FloatType>::init(){
    auto trans = this->transformation_("view")
                 .input("original")
                 .input("rear")
                 .output("result")
                 .types(new TypeClasses::CheckNdimT<FloatType>(1))
                 .types(&ViewRearType::types);
    if(m_fill_value){
        trans.func([](ViewRearType* obj, FunctionArgs& fargs){
                    fargs.args[1];
                });
    }
    else{
        trans.func([](ViewRearType* obj, FunctionArgs& fargs){
                    if(obj->m_fill_value){
                        fargs.args[1];
                    }
                    else {
                        auto& args=fargs.args;
                        auto& ret=fargs.rets[0].x;
                        auto& arg0=args[0].x;
                        size_t head = obj->m_start;
                        if(head){
                            ret.head(head) = arg0.head(head);
                        }
                        size_t tail=ret.size()-obj->m_start-obj->m_len.value();
                        if(tail){
                            ret.tail(tail) = arg0.tail(tail);
                        }
                        args[1];
                        obj->m_fill_value=-1;
                    }
                });

    }

    //TODO: Add GPU option for preallocated buffer
}

template<typename FloatType>
void GNA::GNAObjectTemplates::ViewRearT<FloatType>::types(typename GNAObjectT<FloatType,FloatType>::TypesFunctionArgs& fargs){
    auto& args = fargs.args;

    /// Store the output datatype and check it has enough elements
    const DataType& dt_main = args[0];
    if(dt_main.kind==DataKind::Undefined){
        return;
    }

    if(m_len){
        auto required_length=m_start+m_len.value();
        if(dt_main.shape[0]<required_length)
        {
            throw args.error(dt_main, fmt::format("Transformation {0}: input {1} length should be at least {2}, got {3}",
                                                  args.name(), 0, required_length, dt_main.shape[0]));
        }
    } else {
        if(dt_main.shape[0]<=m_start)
        {
            throw args.error(dt_main, fmt::format("Transformation {0}: input {1} length should be at least {2}, got {3}",
                                                  args.name(), 0, m_start+1, dt_main.shape[0]));
        }
        m_len = dt_main.shape[0]-m_start;
    }


    /// Allocate data if required
    auto& dt_ret = (fargs.rets[0] = dt_main);
    if( !m_data || m_data->type.requiresReallocation(dt_ret) ){
        m_data.reset(new Data<FloatType>(dt_ret));
        if(m_fill_value){
            m_data->x = m_fill_value.value();
        }
    }
    FloatType* buffer_main=m_data->buffer;
    dt_ret.buffer=static_cast<void*>(buffer_main);

    /// Derive the proper datatype for inputs
    DataType dt_sub=dt_ret;
    FloatType* buffer_sub=buffer_main+m_start;
    switch(dt_ret.kind){
        case DataKind::Hist:
            dt_sub.hist().edges(m_len.value()+1, dt_ret.edges.data()+m_start).preallocated(buffer_sub);
            break;
        case DataKind::Points:
            dt_sub.points().shape(m_len.value()).preallocated(buffer_sub);
            break;
        default:
            throw std::runtime_error("something is wrong (ViewRear)");
            break;
    }

    auto& dt_input = args[1];
    if(dt_input.kind==DataKind::Undefined){
        return;
    }

    /// Check that input datatype is consistent
    if(dtypeInconsistent(dt_input, dt_sub)){
        fprintf(stderr, "Main datatype:");
        dt_ret.dump();

        fprintf(stderr, "Sub datatype: ");
        dt_sub.dump();

        fprintf(stderr, "Input datatype: ");
        dt_input.dump();

        throw args.error(dt_input, fmt::format("Transformation {0}: input {1} type is inconsistent with expectations", args.name(), 1));
    }

    /// Check that input data is not preallocatad
    auto& data_in=const_cast<Data<FloatType>&>(args.data(1));
    if(!data_in.allocated && data_in.buffer!=buffer_sub){
        throw args.error(dt_input, fmt::format("Transformation {0}: input {1} may not be preallocated", args.name(), 1));
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

template<typename FloatType>
bool GNA::GNAObjectTemplates::ViewRearT<FloatType>::dtypeInconsistent(const DataType& input, const DataType& required){
    // Check if data types are same (strict)
    if(input==required){
        return false;
    }else if(m_threshold_forbidden){
        return true;
    }

    // If threshold is enabled check if
    // - type is a histogram
    // - types are consisttent
    // - shapes are consistent
    if(input.kind!=DataKind::Hist || input.kind!=required.kind || input.shape!=required.shape){
        return true;
    }

    // Get edges definition
    auto& edges_in = input.edges;
    auto& edges_req= required.edges;

    // Check all, but first and last edges
    if( !equal( edges_in.begin()+1, edges_in.end()-1, edges_req.begin()+1 ) ){
        return true;
    }

    // Check first and last edges
    double check=edges_in[0];
    double left =edges_req[0];
    double right=edges_req[1];
    if(check<left || check>=right){
        return true;
    }

    size_t i=edges_in.size();
    check=edges_in[i-1];
    left =edges_req[i-2];
    right=edges_req[i-1];
    if(check<=left || check>right){
        return true;
    }

    return false;
}

template class GNA::GNAObjectTemplates::ViewRearT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::GNAObjectTemplates::ViewRearT<float>;
#endif
