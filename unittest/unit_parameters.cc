#include "unit_parameters.hh"
#include "taintflag.hh"

bool GNAUnitTest::freeze(changeable* obj){
    try {
        obj->freeze();
    }
    catch(const std::runtime_error& e){
        return true;
    }
    return false;
}

dependant<double> GNAUnitTest::make_test_dependant_double(const std::string& name, size_t size){
    return GNAUnitTest::make_test_dependant<double>(name, size);
}

#ifdef PROVIDE_SINGLE_PRECISION
dependant<float> GNAUnitTest::make_test_dependant_float(const std::string& name, size_t size){
    return GNAUnitTest::make_test_dependant<float>(name, size);
}
#endif

template double GNAUnitTest::debug_fcn_adder<double>();
template void   GNAUnitTest::debug_vfcn_adder<double>(std::vector<double>&);
#ifdef PROVIDE_SINGLE_PRECISION
  template float GNAUnitTest::debug_fcn_adder<float>();
  template void  GNAUnitTest::debug_vfcn_adder<float>(std::vector<float>&);
#endif
