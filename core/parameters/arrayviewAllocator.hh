#pragma once

#include <vector>
#include <boost/noncopyable.hpp>

template<typename T>
class arrayviewAllocator : public boost::noncopyable {
private:
    using allocatorType = arrayviewAllocator<T>;
public:
    virtual ~arrayviewAllocator(){};
    virtual T* allocate(size_t n) = 0;
    virtual const T* data() const noexcept = 0;

    static allocatorType* current() noexcept { return allocatorType::s_current; }
    static void setCurrent(allocatorType* current) noexcept { allocatorType::s_current=current; }
    static void resetCurrent() noexcept { allocatorType::s_current=nullptr; }

private:
    static allocatorType* s_current;
};

template<typename T> arrayviewAllocator<T>* arrayviewAllocator<T>::s_current=nullptr;

template<typename T>
class arrayviewAllocatorSimple : public arrayviewAllocator<T> {
public:
    arrayviewAllocatorSimple(size_t nmax) : m_buffer(nmax), m_current_ptr(m_buffer.data()) {
        m_sizes.reserve(nmax);
    }
    virtual ~arrayviewAllocatorSimple(){};

    T* allocate(size_t n) {
        if(!n){
            return static_cast<T*>(nullptr);
        }
        m_sizes.push_back(n);

        m_size+=n;
        if( m_size>m_buffer.size() ){
            throw std::bad_alloc();
        }
        T* ret = m_current_ptr;
        m_current_ptr += n;
        return ret;
    }

    const T* data() const noexcept { return m_buffer.data(); }
    size_t size() const noexcept { return m_size; }
    size_t maxSize() const noexcept { return m_buffer.size(); }
private:
    size_t m_size=0u;
    std::vector<T> m_buffer;
    std::vector<size_t> m_sizes;
    T* m_current_ptr;
};
