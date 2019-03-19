#include "TreeManager.hh"
#include "arrayviewAllocator.hh"

#include <stdexcept>

template<typename FloatType>
void GNA::TreeManager<FloatType>::update() {

}

template<typename FloatType>
void GNA::TreeManager<FloatType>::resetAllocator() const {
    if(m_allocator){
        if(arrayviewAllocator<FloatType>::current()!=m_allocator){
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
        arrayviewAllocator<FloatType>::setCurrent(m_allocator);
    }
}

template class GNA::TreeManager<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::TreeManager<float>;
#endif
