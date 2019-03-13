#pragma once

#include <stdexcept>
#include <vector>
#include <new>
#include <memory>
#include <algorithm>
#include <complex>

template<typename T>
class arrayviewAllocator {
public:
    virtual ~arrayviewAllocator(){};
    virtual T* allocate(size_t n) = 0;
};

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

template<typename T>
class arrayview {
private:
  using arrayviewType = arrayview<T>;
  using allocatorType = arrayviewAllocator<T>;
public:
  arrayview(T* ptr, size_t size) : m_buffer(ptr), m_size(size) { }

  arrayview(size_t size) : m_allocated(new T[size]), m_buffer(m_allocated.get()), m_size(size) { }
  arrayview(const arrayviewType& other) : arrayview(other.size()) { *this = other; }
  arrayview(const std::initializer_list<T>& other) : arrayview(other.size()) { *this = other; }

  arrayview(size_t size, allocatorType* allocator) : m_allocated(allocator->allocate(size)), m_size(size) { }
  arrayview(const arrayviewType& other, allocatorType* allocator) : arrayview(other.size(), allocator) { *this = other; }
  arrayview(const std::initializer_list<T>& other, allocatorType* allocator) : arrayview(other.size(), allocator) { *this = other; }

  T& operator[](size_t i) noexcept { return m_buffer[i]; }
  const T& operator[](size_t i) const noexcept { return m_buffer[i]; }

  T* data() const noexcept {return m_buffer;}
  T* begin() const noexcept {return m_buffer;}
  T* end() const noexcept {return m_buffer+m_size;}

  size_t size() const noexcept {return m_size;}

  arrayview<T>& operator=(const arrayview<T>& other){ assign(other); return *this; }
  arrayview<T>& operator=(const std::vector<T>& other){ assign(other); return *this; }
  arrayview<T>& operator=(const std::initializer_list<T>& other){ assign(other); return *this; }

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
  void assign(const Other& other) {
    if(m_size!=other.size()){
      throw std::runtime_error("may assign only same size arrays");
    }
    std::copy(other.begin(), other.end(), m_buffer);
  }

  bool isOwner() const noexcept { return (bool)m_allocated; }

  std::complex<T>& complex() noexcept { return *reinterpret_cast<std::complex<T>*>(m_buffer); }
  const std::complex<T>& complex() const noexcept { return *reinterpret_cast<std::complex<T>*>(m_buffer); }
private:
  std::unique_ptr<T> m_allocated{nullptr};
  T*                 m_buffer;
  size_t             m_size;
};

