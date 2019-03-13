#pragma once

#include <vector>
#include <memory>
#include <algorithm>

template<typename T>
class arrayview {
private:
  using arrayviewType = arrayview<T>;
public:
  arrayview(T* ptr, size_t size) : m_buffer(ptr), m_size(size) { }
  arrayview(size_t size) : m_allocated(new T[size]), m_buffer(m_allocated.get()), m_size(size) { }
  arrayview(const arrayviewType& other) : arrayview(other.size()) { *this = other; }
  arrayview(const std::initializer_list<T>& other) : arrayview(other.size()) { *this = other; }

  T& operator[](size_t i) noexcept { return m_buffer[i]; }
  const T& operator[](size_t i) const noexcept { return m_buffer[i]; }

  T* data() const noexcept {return m_buffer;}
  T* begin() const noexcept {return m_buffer;}
  T* end() const noexcept {return m_buffer+m_size;}

  size_t size() const noexcept {return m_size;}

  arrayview<T>& operator=(const arrayview<T>& other){ assign(other); return *this; }
  arrayview<T>& operator=(const std::vector<T>& other){ assign(other); return *this; }
  arrayview<T>& operator=(const std::initializer_list<T>& other){ assign(other); return *this; }

  bool operator==(const arrayview<T>& other){ return equal(other); }
  bool operator==(const std::vector<T>& other){ return equal(other); }
  bool operator==(const std::initializer_list<T>& other){ return equal(other); }

  bool operator!=(const arrayview<T>& other){ return notequal(other); }
  bool operator!=(const std::vector<T>& other){ return notequal(other); }
  bool operator!=(const std::initializer_list<T>& other){ return notequal(other); }

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

  bool isOwner() { return (bool)m_allocated; }
private:
  std::unique_ptr<T> m_allocated{nullptr};
  T*                 m_buffer;
  size_t             m_size;
};

