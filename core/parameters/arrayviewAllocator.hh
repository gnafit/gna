#pragma once

#include <vector>

template<typename T>
class arrayviewAllocator {
private:
    using allocatorType = arrayviewAllocator<T>;
public:
    virtual ~arrayviewAllocator(){};
    virtual T* allocate(size_t n) = 0;

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
    arrayviewAllocatorSimple(size_t nmax) : m_buffer(nmax), m_current_ptr(m_buffer.data()) { }
    virtual ~arrayviewAllocatorSimple(){};

    T* allocate(size_t n) {
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
    T* m_current_ptr;
};
