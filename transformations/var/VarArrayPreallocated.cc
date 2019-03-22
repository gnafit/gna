#include "VarArrayPreallocated.hh"
#include "arrayviewAllocator.hh"
#include <stdexcept>

using std::next;

//template<typename FloatType>
//GNA::GNAObjectTemplates::VarArrayPreallocatedT<FloatType>::VarArrayPreallocatedT(const std::vector<std::string>& varnames) :
//m_vars(varnames.size())
//{
//}

template<typename FloatType>
GNA::GNAObjectTemplates::VarArrayPreallocatedT<FloatType>::VarArrayPreallocatedT(const std::vector<variable<FloatType>>& vars) :
m_vars(vars.size())
{
    m_allocator = arrayviewAllocator<FloatType>::current();
    if(!m_allocator){
        throw std::runtime_error("VarArrayPreallocated expects the allocator to be defined");
    }
    const auto* root=m_allocator->data();

    for (size_t i = 0; i < vars.size(); ++i) {
        auto& var   = vars[i];
        if(var.root()!=root){
            throw std::runtime_error(fmt::format("Variable {} is initialized outside of the current pool", var.name()));
        }
        auto& field = m_vars[i];
        auto handle=this->variable_(&field, var.name());
        handle.bind(var);
        if(var.hasFunc()){
            m_dependants.push_back(field);
        }
    }

    initTransformation();
    variablesBound();
}

template<typename FloatType>
void GNA::GNAObjectTemplates::VarArrayPreallocatedT<FloatType>::initTransformation(){
    this->transformation_("vararray")
         .output("points")
         .types(&VarArrayPreallocatedType::typesFunction)
         .func(&VarArrayPreallocatedType::function)
         .finalize();
}

template<typename FloatType>
void GNA::GNAObjectTemplates::VarArrayPreallocatedT<FloatType>::typesFunction(VarArrayPreallocatedT<FloatType>::TypesFunctionArgs& fargs){
    fargs.rets[0].points().shape(m_allocator->size()).preallocated(m_allocator->data());
}

template<typename FloatType>
void GNA::GNAObjectTemplates::VarArrayPreallocatedT<FloatType>::function(VarArrayPreallocatedT<FloatType>::FunctionArgs& fargs){
    for(auto& var : m_dependants){
        var.update();
    }
}

template class GNA::GNAObjectTemplates::VarArrayPreallocatedT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::GNAObjectTemplates::VarArrayPreallocatedT<float>;
#endif
