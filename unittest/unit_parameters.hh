#pragma once

#include "dependant.hh"

namespace GNAUnitTest{
    bool freeze(changeable* obj);

    template<typename FloatType>
    dependant<FloatType> make_test_dependant(const std::string& name, size_t size=1u);

    dependant<double> make_test_dependant_double(const std::string& name, size_t size=1u);
#ifdef PROVIDE_SINGLE_PRECISION
    dependant<float> make_test_dependant_float(const std::string& name, size_t size=1u);
#endif

    template<typename FloatType>
    FloatType debug_fcn_adder();

    template<typename FloatType>
    void debug_vfcn_adder(arrayview<FloatType>& dest);
}

template<typename FloatType>
FloatType GNAUnitTest::debug_fcn_adder(){
  static FloatType ret=0.0;
  ++ret;
  return ret;
}

template<typename FloatType>
void GNAUnitTest::debug_vfcn_adder(arrayview<FloatType>& dest){
  static FloatType ret=0.0;
  for (size_t i = 0; i < dest.size(); ++i) {
    dest[i] = ++ret;
  }
}

template<typename FloatType>
dependant<FloatType> GNAUnitTest::make_test_dependant(const std::string& name, size_t size){
    if(size==1u){
        return dependant<FloatType>(debug_fcn_adder<FloatType>, {}, name.c_str(), size);
    }
    return dependant<FloatType>(debug_vfcn_adder<FloatType>, {}, name.c_str(), size);
}
