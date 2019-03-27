#pragma once

#include <vector>
#include <string>
#include <initializer_list>
#include "variable.hh"

template<typename SourceFloatType,typename SinkFloatType>
class GNAObjectT;

namespace TransformationTypes{
    template<typename SourceFloatType, typename SinkFloatType>
        class EntryT;
}

namespace ParametrizedTypes{
    class ParametrizedBase;
}

namespace GNA{
    template<typename FloatType> class TreeManager;
}

namespace TransformationTypes{
    template<typename FloatType,typename SizeType=unsigned int>
    class GPUVariables {
    protected:
        using TreeManagerType = GNA::TreeManager<FloatType>;
    public:
        using TransformationType = TransformationTypes::EntryT<FloatType,FloatType>;

        GPUVariables(TransformationType* entry);

        ~GPUVariables() {
            deAllocateDevice();
        };

        void allocateDevice();
        void deAllocateDevice();

        void provideSignatureHost(SizeType &nvars, FloatType** &values);
        void provideSignatureDevice(SizeType &nvars, FloatType** &values);

        void dump(const std::string& type);

        void readVariables(ParametrizedTypes::ParametrizedBase* parbase);

        void syncHost2Device();
    private:
        std::vector<variable<FloatType>> m_variables;
        TreeManagerType* m_tmanager;

        std::vector<FloatType*>  h_value_pointers_host;
        std::vector<FloatType*>  h_value_pointers_dev;

        FloatType**              d_value_pointers_dev{nullptr};
    };

    template<typename FloatType,typename SizeType>
        void GPUVariables<FloatType,SizeType>::deAllocateDevice(){
            if(d_value_pointers_dev){
                /// TODO
                d_value_pointers_dev=nullptr;
            }
        }

    template<typename FloatType,typename SizeType>
        void GPUVariables<FloatType,SizeType>::allocateDevice(){
            /// h_value_pointers_dev -> d_value_pointers_dev
            //
        }

    template<typename FloatType,typename SizeType>
        void GPUVariables<FloatType,SizeType>::dump(const std::string& type){
            size_t nvars=h_value_pointers_host.size();

            printf("Dumping GPUVariables (global) of size %zu", nvars);
            if(type.size()){
                printf(" (%s)", type.c_str());
            }
            printf("\n");
            for (size_t i = 0; i < nvars; ++i) {
                auto value=*h_value_pointers_host[i];
                auto name=m_variables[i].name();

                printf("  Variable %zu: %12.6g    %s\n", i, value, name);
            }
        }

    template<typename FloatType,typename SizeType>
        void GPUVariables<FloatType,SizeType>::provideSignatureHost(SizeType &nvars, FloatType** &values){
            nvars=h_value_pointers_host.size();
            values=h_value_pointers_host.data();
        }

    template<typename FloatType,typename SizeType>
        void GPUVariables<FloatType,SizeType>::provideSignatureDevice(SizeType &nvars, FloatType** &values){
            nvars=h_value_pointers_dev.size();
            values=d_value_pointers_dev;
        }
}
