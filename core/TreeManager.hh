#pragma once

#include <boost/noncopyable.hpp>
#include <boost/pool/object_pool.hpp>

namespace GNA{
    template <class T>
    class ChangeableAllocator;

    template<typename FloatType>
    class TreeManager : public boost::noncopyable
    {
    protected:
        using TreeManagerType = TreeManager<FloatType>;
        using PoolType = boost::object_pool<FloatType>;
        using AllocatorType = ChangeableAllocator<FloatType>;

    public:
        TreeManager() : m_pool(m_max_size, m_max_size), m_allocator(&m_pool) {
        }
        virtual ~TreeManager(){

        }

    protected:
        const size_t m_max_size=1000;
        PoolType m_pool;
        AllocatorType m_allocator;
        /* data */
    };

    template <class FloatType>
    class ChangeableAllocator {
    private:
        using PoolType = boost::object_pool<FloatType>;

    public:
        using value_type = FloatType;

        ChangeableAllocator(PoolType* pool) : pool(pool) {

        };

        FloatType* allocate(std::size_t n) {
            FloatType *ptr0{nullptr},ptr;
            for (size_t i = 0; i < n; ++i) {
                ptr=pool->malloc();
                if(ptr==nullptr){
                    throw std::bad_alloc();
                }
                if(ptr0==nullptr){
                    ptr0=ptr;
                }
            }
            return ptr0;
        }

    private:
        PoolType* pool;
    };
}
