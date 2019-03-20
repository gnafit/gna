#include "VarArray.hh"

using std::next;

template<typename FloatType>
GNA::GNAObjectTemplates::VarArrayT<FloatType>::VarArrayT(const std::vector<std::string>& varnames) :
m_vars(varnames.size())
{
    for (size_t i = 0; i < varnames.size(); ++i) {
        this->variable_(&m_vars[i], varnames[i]);
    }

    this->transformation_("vararray")                           /// Initialize the transformation points.
         .output("points")
         .types(&VarArrayType::typesFunction)
         .func(&VarArrayType::function)
         .finalize();                                            /// Tell the initializer that there are no more configuration and it may initialize the types.
}

template<typename FloatType>
void GNA::GNAObjectTemplates::VarArrayT<FloatType>::typesFunction(VarArrayT<FloatType>::TypesFunctionArgs& fargs){
    fargs.rets[0] = DataType().points().shape(m_vars.size());
}

template<typename FloatType>
void GNA::GNAObjectTemplates::VarArrayT<FloatType>::function(VarArrayT<FloatType>::FunctionArgs& fargs){
    auto* buffer = fargs.rets[0].buffer;
    for( auto& var : m_vars ){
        *buffer = var.value();
        buffer=next(buffer);
    }
}

template class GNA::GNAObjectTemplates::VarArrayT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::GNAObjectTemplates::VarArrayT<float>;
#endif
