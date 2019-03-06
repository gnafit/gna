#pragma once

#include <vector>
#include <string>
#include <initializer_list>
#include "variable.hh"

namespace TransformationTypes{
    template<typename FloatType,typename SizeType=unsigned int>
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

        template<typename Container>
        void readVariables(Container& list);

        void syncHost2Device();
    private:
        void readVariable(size_t i, const variable<FloatType>& var);

        std::vector<std::string> names;
        std::vector<FloatType>   h_values;
        std::vector<FloatType*>  h_value_pointers_host;
        std::vector<FloatType*>  h_value_pointers_dev;

        FloatType*               d_values{nullptr};
        FloatType**              d_value_pointers_dev{nullptr};
    };

    template<typename FloatType,typename SizeType>
    void GPUVariablesLocal<FloatType,SizeType>::setSize(size_t size){
        if(names.size()==size){
            return;
        }
        deAllocateDevice();

        names.resize(size);
        h_values.resize(size);
        h_value_pointers_host.resize(size);
        h_value_pointers_dev.resize(size);

        auto* ptr=h_values.data();
        for (size_t i = 0; i < h_value_pointers_host.size(); ++i) {
            h_value_pointers_host[i]=ptr;
            std::advance(ptr, 1);
        }

        allocateDevice();
    }

    template<typename FloatType,typename SizeType>
    void GPUVariablesLocal<FloatType,SizeType>::deAllocateDevice(){
        if(d_values){
            /// TODO
            d_values=nullptr;
        }
        if(d_value_pointers_dev){
            /// TODO
            d_value_pointers_dev=nullptr;
        }
    }

    template<typename FloatType,typename SizeType>
    void GPUVariablesLocal<FloatType,SizeType>::allocateDevice(){
        /// allocate d_values (same as h_values, no sync is needed here)
        auto* ptr=d_values;
        for (size_t i = 0; i < h_value_pointers_dev.size(); ++i) {
            h_value_pointers_dev[i]=ptr;
            std::advance(ptr, 1);
        }
        /// h_value_pointers_dev -> d_value_pointers_dev
    }

    template<typename FloatType,typename SizeType>
    void GPUVariablesLocal<FloatType,SizeType>::dump(const std::string& type){
        size_t nvars=h_values.size();

        printf("Dumping GPUVariablesLocal of size %zu", nvars);
        if(type.size()){
            printf(" (%s)", type.c_str());
        }
        printf("\n");
        for (size_t i = 0; i < nvars; ++i) {
            auto  value=*h_value_pointers_host[i];
            auto& name=names[i];

            printf("  Variable %zu: %12.6g    %s\n", i, value, name.c_str());
        }
    }

    template<typename FloatType,typename SizeType>
    void GPUVariablesLocal<FloatType,SizeType>::provideSignatureHost(SizeType &nvars, FloatType** &values){
        nvars=h_value_pointers_host.size();
        values=h_value_pointers_host.data();
    }

    template<typename FloatType,typename SizeType>
    void GPUVariablesLocal<FloatType,SizeType>::provideSignatureDevice(SizeType &nvars, FloatType** &values){
        nvars=h_value_pointers_dev.size();
        values=d_value_pointers_dev;
    }

    template<typename FloatType,typename SizeType>
    template<typename Container>
    void GPUVariablesLocal<FloatType,SizeType>::readVariables(Container& vars){
        setSize(vars.size());
        size_t i(0);
        for (auto& var: vars) {
            readVariable(i, var);
            ++i;
        }
        syncHost2Device();
    }

    template<typename FloatType,typename SizeType>
    void GPUVariablesLocal<FloatType,SizeType>::readVariable(size_t i, const variable<FloatType>& var){
        names[i]=var.name();
        h_values[i]=var.value();
    }

    template<typename FloatType,typename SizeType>
    void GPUVariablesLocal<FloatType,SizeType>::syncHost2Device(){
        /// h_values -> d_values
    }
}
