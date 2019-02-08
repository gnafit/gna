#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <iostream>
#include "TransformationEntry.hh"
#include "GPUVariablesLocal.hh"
#include "GPUFunctionData.hh"

#include "config_vars.h"

#ifdef GNA_CUDA_SUPPORT
#include "GpuBasics.hh"
#endif

namespace TransformationTypes{
    template<typename FloatType, typename SizeType=unsigned int>
    class GPUFunctionArgsT {
    public:
        using EntryType = EntryT<FloatType,FloatType>;
        GPUFunctionArgsT(EntryType* entry) : m_entry(entry){

        }

        ~GPUFunctionArgsT(){

        }

        template<typename Container>
        void readVariables(Container& vars){ 
		m_vars.readVariables(vars);
		m_entry->setEntryDataLocation(DataLocation::Device);
	}

        void updateTypesHost();
        void updateTypesDevice();
        void updateTypes() {
		updateTypesHost();
		if (m_entry->getEntryLocation() == DataLocation::Device){
		    updateTypesDevice();
		}
	}
        void provideSignatureHost();
        void provideSignatureDevice();
        void dump();

        SizeType    nvars{0u};       ///< number of variables
        FloatType **vars{nullptr};   ///< list of pointers to variable values
        SizeType    nargs{0u};       ///< number of args
        FloatType **args{nullptr};   ///< list of pointers to args
        SizeType  **argshapes{0u};   ///< list of pointers to shapes of args
        SizeType    nrets{0u};       ///< number of rets
        FloatType **rets{nullptr};   ///< list of pointers to rets
        SizeType  **retshapes{0u};   ///< list of pointers to shapes of rets
        SizeType    nints{0u};       ///< number of ints
        FloatType **ints{nullptr};   ///< list of pointers to ints
        SizeType  **intshapes{0u};   ///< list of pointers to shapes of ints

    private:
        EntryType* m_entry;

        GPUVariablesLocal<FloatType,SizeType> m_vars; ///< Handler for variables (local)
        GPUFunctionData<FloatType,SizeType>   m_args; ///< Handler for inputs
        GPUFunctionData<FloatType,SizeType>   m_rets; ///< Handler for outputs
        GPUFunctionData<FloatType,SizeType>   m_ints; ///< Handler for storages
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
#ifdef GNA_CUDA_SUPPORT
        m_args.fillContainersDevice(m_entry->sources);
        m_rets.fillContainersDevice(m_entry->sinks);
        m_ints.fillContainersDevice(m_entry->storages);
#else
	std::cerr << "There is no CUDA support, so I can't switch your function to GPU-based one." << std::endl;
#endif

        provideSignatureDevice();
    }

    template<typename FloatType,typename SizeType>
    void GPUFunctionArgsT<FloatType,SizeType>::provideSignatureHost(){
        m_vars.provideSignatureHost(nvars, vars);
        m_args.provideSignatureHost(nargs, args, argshapes);
        m_rets.provideSignatureHost(nrets, rets, retshapes);
        m_ints.provideSignatureHost(nints, ints, intshapes);
    }

    template<typename FloatType,typename SizeType>
    void GPUFunctionArgsT<FloatType,SizeType>::provideSignatureDevice(){
        m_vars.provideSignatureDevice(nvars, vars);
        m_args.provideSignatureDevice(nargs, args, argshapes);
        m_rets.provideSignatureDevice(nrets, rets, retshapes);
        m_ints.provideSignatureDevice(nints, ints, intshapes);
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

    using GPUFunctionArgs = GPUFunctionArgsT<double>;
}
