#pragma once

#include <vector>
#include <string>
#include <initializer_list>
#include "variable.hh"

#include <iostream>
#include "config_vars.h"
#ifdef GNA_CUDA_SUPPORT
#include "GpuBasics.hh"
#endif

namespace ParametrizedTypes{
  class ParametrizedBase;
}

namespace TransformationTypes{
    template<typename FloatType,typename SizeType>
    class GPUVariablesLocal {
    public:
        GPUVariablesLocal() {

        };

        ~GPUVariablesLocal() {
            deAllocateDevice();
        };

        void setSize(size_t size);

        void allocateDevice();
        void deAllocateDevice();

        void provideSignatureHost(SizeType &nvars, FloatType** &values);
        void provideSignatureDevice(SizeType &nvars, FloatType** &values);

        void dump(const std::string& type);

        void readVariables();
        void readVariables(ParametrizedTypes::ParametrizedBase* parbase);

        void syncHost2Device();
    private:
        void readVariable(size_t i, const variable<FloatType>& var);

        std::vector<variable<FloatType>> m_variables;

        std::vector<std::string> names;
        std::vector<FloatType>   h_values;
        std::vector<FloatType*>  h_value_pointers_host;
        std::vector<FloatType*>  h_value_pointers_dev;

        FloatType*               d_values{nullptr};
        FloatType**              d_value_pointers_dev{nullptr};
    };
}
