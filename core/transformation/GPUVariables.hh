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
    template<typename FloatType,typename SizeType>
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
}
