#pragma once

#include <stdexcept>
#include <new>
#include <memory>
#include <algorithm>
#include <complex>
#include "arrayviewAllocator.hh"

template<typename T>
class arrayview {
private:
    using arrayviewType = arrayview<T>;
    using allocatorType = arrayviewAllocator<T>;
public:
    arrayview(T* ptr, size_t size) : m_buffer(ptr), m_size(size), m_root(ptr), m_offset(0u) { }

    arrayview(size_t size) : m_size(size) {
        if(!size){
            return;
        }
        auto* allocator = allocatorType::current();
        if(allocator){
            m_buffer = allocator->allocate(size);
            m_root   = allocator->data();
            m_offset = m_buffer - m_root;
        }
        else{
            m_buffer = new T[size];
            m_allocated.reset(m_buffer);
            m_root = m_buffer;
        }
    }
    arrayview(const arrayviewType& other) : arrayview(other.size()) { *this = other; }
    arrayview(const std::initializer_list<T>& other) : arrayview(other.size()) { *this = other; }
    arrayview(const std::complex<T>& other) : arrayview(2) { this->complex() = other; }

    arrayview(size_t size, allocatorType* allocator) : m_buffer(allocator->allocate(size)), m_size(size) {
        m_root   = allocator->data();
        m_offset = m_buffer - m_root;
    }
    arrayview(const arrayviewType& other, allocatorType* allocator) : arrayview(other.size(), allocator) { *this = other; }
    arrayview(const std::initializer_list<T>& other, allocatorType* allocator) : arrayview(other.size(), allocator) { *this = other; }

    T& operator[](size_t i) noexcept { return m_buffer[i]; }
    const T& operator[](size_t i) const noexcept { return m_buffer[i]; }

    T* data() const noexcept {return m_buffer;}
    T* begin() const noexcept {return m_buffer;}
    T* end() const noexcept {return m_buffer+m_size;}

    size_t size() const noexcept {return m_size;}

    arrayview<T>& operator=(const arrayview<T>& other){ copyfrom(other); return *this; }
    arrayview<T>& operator=(const std::vector<T>& other){ copyfrom(other); return *this; }
    arrayview<T>& operator=(const std::initializer_list<T>& other){ copyfrom(other); return *this; }

    bool operator==(const arrayview<T>& other) const noexcept { return equal(other); }
    bool operator==(const std::vector<T>& other) const noexcept { return equal(other); }
    bool operator==(const std::initializer_list<T>& other) const noexcept { return equal(other); }

    bool operator!=(const arrayview<T>& other) const noexcept { return notequal(other); }
    bool operator!=(const std::vector<T>& other) const noexcept { return notequal(other); }
    bool operator!=(const std::initializer_list<T>& other) const noexcept { return notequal(other); }

    template<class Other>
        bool equal(const Other& other) const noexcept {
            if(m_size!=other.size()){
                return false;
            }
            return std::equal(begin(), end(), other.begin());
        }

    template<class Other>
        bool notequal(const Other& other) const noexcept {
            if(m_size!=other.size()){
                return true;
            }
            return !std::equal(begin(), end(), other.begin());
        }

    template<class Other>
        void copyfrom(const Other& other) {
            if(m_size!=other.size()){
                throw std::runtime_error("may assign only same size arrays");
            }
            std::copy(other.begin(), other.end(), m_buffer);
        }

    bool isOwner() const noexcept { return (bool)m_allocated; }

    std::complex<T>& complex() noexcept { return *reinterpret_cast<std::complex<T>*>(m_buffer); }
    const std::complex<T>& complex() const noexcept { return *reinterpret_cast<std::complex<T>*>(m_buffer); }

    const T* root() const noexcept { return m_root; }
    size_t offset() const noexcept { return m_offset; }
private:
    std::unique_ptr<T> m_allocated{nullptr};
    T*                 m_buffer{nullptr};
    size_t             m_size{0u};
    const T*           m_root{nullptr};
    size_t             m_offset{0u};
};

