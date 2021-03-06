#include "TreeManager.hh"
#include "arrayviewAllocator.hh"
#include "VarArrayPreallocated.hh"

#include <stdexcept>

template<typename FloatType>
GNA::TreeManager<FloatType>::TreeManager(size_t allocatepars) {
    if (allocatepars) {
        m_allocator.reset(new arrayviewAllocatorSimple<FloatType>(allocatepars));
    }
}

template<typename FloatType>
GNA::TreeManager<FloatType>::~TreeManager() { }

template<typename FloatType>
void GNA::TreeManager<FloatType>::setVariables(const std::vector<variable<FloatType>>& variables) {
    m_vararray.reset(new VarArrayType(variables));
    m_transformation.reset(new TransformationDescriptorType(m_vararray->transformations.front()));
    m_output.reset(new OutputDescriptorType(m_transformation->outputs.front()));

    #ifdef GNA_CUDA_SUPPORT
    for(auto& trans: m_transformations){
        if(trans->getEntryLocation()==DataLocation::Device){
            m_output->requireGPU();
            break;
        }
    }
    #endif

    for(auto& trans: m_transformations){
        trans->functionargs->readVariables();
    }
}

template<typename FloatType>
bool GNA::TreeManager<FloatType>::hasVariable(const variable<void>& variable) {
    if(!m_vararray){
        return false;
    }

    return m_vararray->hasVariable(variable);
}

template<typename FloatType>
bool GNA::TreeManager<FloatType>::consistentVariable(const variable<FloatType>& variable) {
    if(!m_allocator){
        return false;
    }

    return variable.values().root() == m_allocator->data();
}

template<typename FloatType>
void GNA::TreeManager<FloatType>::update() {
    // Caution: triggering touch_global() may cause infinite loop.
    if(m_transformation){
#ifdef GNA_CUDA_SUPPORT
        auto& data=(*m_transformation)[0];
        if (data.gpuArr){
            data.gpuArr->sync(DataLocation::Device);
        }
#else
        m_transformation->touch_local();
#endif
    }
}

template<typename FloatType>
void GNA::TreeManager<FloatType>::makeCurrent() {
    if(GNA::TreeManager<FloatType>::current()){
        throw std::runtime_error("Unable to set treemanager. Another instance is already set.");
    }
    GNA::TreeManager<FloatType>::setCurrent(this);

    setAllocator();
}

template<typename FloatType>
void GNA::TreeManager<FloatType>::resetAllocator() const {
    if(m_allocator){
        if(arrayviewAllocator<FloatType>::current()!=m_allocator.get()){
            throw std::runtime_error("Unable to reset allocator. Another allocator is currently active.");
        }
        arrayviewAllocator<FloatType>::resetCurrent();
    }
}

template<typename FloatType>
void GNA::TreeManager<FloatType>::setAllocator() const {
    if(arrayviewAllocator<FloatType>::current()){
        throw std::runtime_error("Unable to set allocator. Another allocator is currently active.");
    }
    if(m_allocator){
        arrayviewAllocator<FloatType>::setCurrent(m_allocator.get());
    }
}

template class GNA::TreeManager<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::TreeManager<float>;
#endif
