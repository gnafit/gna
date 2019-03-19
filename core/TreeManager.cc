#include "TreeManager.hh"
#include "arrayviewAllocator.hh"

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
void GNA::TreeManager<FloatType>::update() {
    printf("update");
}

template<typename FloatType>
void GNA::TreeManager<FloatType>::makeCurrent() {
    setAllocator();

    if(GNA::TreeManager<FloatType>::current()){
        throw std::runtime_error("Unable to set treemanager. Another instance is already set.");
    }
    GNA::TreeManager<FloatType>::setCurrent(this);
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
