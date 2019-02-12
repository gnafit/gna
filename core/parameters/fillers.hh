#pragma once

#include <array>
#include "variable.hh"
#include "parameter.hh"

template <typename FloatType, size_t size=1u>
size_t copyVariableToArray(const variable<FloatType>& var, FloatType* dest) {
  static_assert(size==1u, "copyVariableToArray<> should be of size 1 for double/float/etc variables");
  *dest = var.value();
  return 1u;
}

template <typename FloatType, size_t size>
size_t copyVariableToArray(const variable<std::array<FloatType,size>>& var, FloatType* dest) {
  const auto& values=var.value();
  for(auto value: values){
    *dest = value;
    std::advance(dest,1);
  }
  return values.size();
}

template <typename FloatType, size_t size=1u>
size_t copyArrayToParameter(FloatType* source, parameter<FloatType>& par) {
  static_assert(size==1u, "copyArrayToParameter<> should be of size 1 for double/float/etc parameters");
  par.set(*source);
  return 1u;
}

template <typename FloatType, size_t size>
size_t copyArrayToParameter(FloatType* source, parameter<std::array<FloatType,size>>& par) {
  std::array<FloatType,size> array;
  std::copy(source, std::next(source,size), array.begin());
  par.set(array);
  return size;
}

