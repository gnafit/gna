#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <iostream>
#include "TransformationEntry.hh"
#include "GPUVariablesLocal.hh"
#include "GPUVariables.hh"
#include "GPUFunctionData.hh"

#include "config_vars.h"

#ifdef GNA_CUDA_SUPPORT
#include "GpuBasics.hh"
#include "DataLocation.hh"
#endif

namespace TransformationTypes{
    template<typename FloatType, typename SizeType=size_t>
    class GPUFunctionArgsT {
    public:
        using EntryType = EntryT<FloatType,FloatType>;
        GPUFunctionArgsT(EntryType* entry) : m_entry(entry), m_vars_global(entry) {

        }

        ~GPUFunctionArgsT(){
                /// deallocate m_argsmapping_dev
        }

        void readVariables(ParametrizedTypes::ParametrizedBase* parbase){
            m_vars.readVariables(parbase);
            m_vars_global.readVariables(parbase);
            //setAsDevice();
        }

        void readVariablesLocal(){
            m_vars.readVariables();
            //setAsDevice();
        }

        //void setAsDevice() {
//#ifdef GNA_CUDA_SUPPORT
            //m_entry->setEntryDataLocation(DataLocation::Device);
//#endif
        //}

        //void setAsHost() {
//#ifdef GNA_CUDA_SUPPORT
            //m_entry->setEntryDataLocation(DataLocation::Host);
//#endif
        //}

        void updateTypesHost();
        void updateTypesDevice();

        void updateTypes() {
            updateTypesHost();
#ifdef GNA_CUDA_SUPPORT
            if (m_entry->getEntryLocation() == DataLocation::Device){
                updateTypesDevice();
            }
#endif
        }
        void provideSignatureHost(bool local=false);
        void provideSignatureDevice(bool local=false);
        void dump();

        SizeType    nvars{0u};            ///< number of variables
        FloatType **vars{nullptr};        ///< list of pointers to variable values
        SizeType    nargs{0u};            ///< number of args
        FloatType **args{nullptr};        ///< list of pointers to args
        SizeType  **argshapes{nullptr};   ///< list of pointers to shapes of args
        SizeType   *argsmapping{nullptr}; ///< array mapping each of the args to specific ret
        SizeType    nrets{0u};            ///< number of rets
        FloatType **rets{nullptr};        ///< list of pointers to rets
        SizeType  **retshapes{nullptr};   ///< list of pointers to shapes of rets
        SizeType    nints{0u};            ///< number of ints
        FloatType **ints{nullptr};        ///< list of pointers to ints
        SizeType  **intshapes{nullptr};   ///< list of pointers to shapes of ints


    private:
        EntryType* m_entry;

        GPUVariablesLocal<FloatType,SizeType> m_vars; ///< Handler for variables (local)
        GPUVariables<FloatType,SizeType>      m_vars_global; ///< Handler for variables (local)
        GPUFunctionData<FloatType,SizeType>   m_args; ///< Handler for inputs
        GPUFunctionData<FloatType,SizeType>   m_rets; ///< Handler for outputs
        GPUFunctionData<FloatType,SizeType>   m_ints; ///< Handler for storages

        SizeType* m_argsmapping_dev{nullptr};
    };
}
