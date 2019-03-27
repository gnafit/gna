#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <iostream>
#include "TransformationEntry.hh"
#include "GPUVariablesLocal.hh"
#include "GPUVariables.hh"
#include "GPUFunctionData.hh"

namespace TransformationTypes{
    template<typename FloatType, typename SizeType=size_t>
    class GPUFunctionArgsT {
    public:
        using EntryType = EntryT<FloatType,FloatType>;
        GPUFunctionArgsT(EntryType* entry) : m_entry(entry), m_vars_global(entry) {

        }

        ~GPUFunctionArgsT(){
            ///TODO: deallocate m_argsmapping_dev
        }

        void readVariables(ParametrizedTypes::ParametrizedBase* parbase){
            m_vars.readVariables(parbase);
            m_vars_global.readVariables(parbase);
        }

        void readVariablesLocal(){
            m_vars.readVariables();
        }

        void updateTypesHost();
        void updateTypesDevice();
        void updateTypes() { updateTypesHost(); }
        void provideSignatureHost();
        void provideSignatureDevice();
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


    template<typename FloatType,typename SizeType>
    void GPUFunctionArgsT<FloatType,SizeType>::updateTypesHost(){
        m_args.fillContainers(m_entry->sources);
        m_rets.fillContainers(m_entry->sinks);
        m_ints.fillContainers(m_entry->storages);

        provideSignatureHost();
    }

    template<typename FloatType,typename SizeType>
    void GPUFunctionArgsT<FloatType,SizeType>::updateTypesDevice(){
        ///TODO: deallocate m_argsmapping_dev

        m_args.fillContainers(m_entry->sources);
        m_rets.fillContainers(m_entry->sinks);
        m_ints.fillContainers(m_entry->storages);

        ///TODO: allocate m_argsmapping_dev
        ///TODO: sync m_entry->mapping to m_argsmapping_dev

        provideSignatureDevice();
    }

    template<typename FloatType,typename SizeType>
    void GPUFunctionArgsT<FloatType,SizeType>::provideSignatureHost(){
        m_vars.provideSignatureHost(nvars, vars);
        m_args.provideSignatureHost(nargs, args, argshapes);
        m_rets.provideSignatureHost(nrets, rets, retshapes);
        m_ints.provideSignatureHost(nints, ints, intshapes);

        argsmapping = m_entry->mapping.size() ? m_entry->mapping.data() : nullptr;
    }

    template<typename FloatType,typename SizeType>
    void GPUFunctionArgsT<FloatType,SizeType>::provideSignatureDevice(){
        m_vars.provideSignatureDevice(nvars, vars);
        m_args.provideSignatureDevice(nargs, args, argshapes);
        m_rets.provideSignatureDevice(nrets, rets, retshapes);
        m_ints.provideSignatureDevice(nints, ints, intshapes);

        argsmapping = m_argsmapping_dev;
    }

    template<typename FloatType,typename SizeType>
    void GPUFunctionArgsT<FloatType,SizeType>::dump(){
        printf("Dumping GPU args state\n");

        m_vars.dump("variables");
        printf("\n");

        m_args.dump("sources");
        printf("\n");

        m_rets.dump("sinks");
        printf("\n");

        m_ints.dump("storages");
        printf("\n");

    }
}
