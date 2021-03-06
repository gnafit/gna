#include "GPUVariables.hh"
#include "ParametrizedBase.hh"
#include "TreeManager.hh"
#include "TransformationEntry.hh"
#include "TransformationDescriptor.hh"

#include "config_vars.h"
#ifdef GNA_CUDA_SUPPORT
#include "GpuBasics.hh"
#endif

using GNA::TreeManager;

template<typename FloatType,typename SizeType>
TransformationTypes::GPUVariables<FloatType,SizeType>::GPUVariables(TransformationTypes::EntryT<FloatType,FloatType>* transformation){
    m_tmanager = TreeManager<FloatType>::current();
    if(!m_tmanager){
        return;
    }

    if(m_tmanager!=transformation->m_tmanager){
        throw std::runtime_error("Transformation is not managed by the current TreeManager");
    }
//    std::cout << "constructor " << __PRETTY_FUNCTION__ << " h_value_pointers_host size = " << h_value_pointers_host.size() <<
//             " h_value_pointers_dev size = " << h_value_pointers_dev.size() << " " << (void*)this << std::endl;
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariables<FloatType,SizeType>::readVariables(ParametrizedTypes::ParametrizedBase* parbase){
    if(!m_tmanager){
        return;
    }
    m_variables.clear();
    for(auto& entry: parbase->m_entries){
        auto& var=entry.var;
        if(!var.istype<FloatType>()){
            throw std::runtime_error("GPUVariables: Invalid variable type");
        }
        variable<FloatType> fvar(var);
        if(!m_tmanager->consistentVariable(fvar)){
            throw std::runtime_error("The variable is inconsistent with the current TreeManager");
        }
        m_variables.emplace_back(fvar);
    }

    auto size=m_variables.size();
    h_value_pointers_host.resize(size);
    h_value_pointers_dev.resize(size);

//    std::cout << "readVars "<< __PRETTY_FUNCTION__ << " h_value_pointers_host size = " << h_value_pointers_host.size() <<
//             " h_value_pointers_dev size = " << h_value_pointers_dev.size() << " " << (void*)this << std::endl;
    deAllocateDevice();
    #ifdef GNA_CUDA_SUPPORT
    OutputHandleT<FloatType>* output = m_tmanager->getOutput();
    if(!output){
        return;
        //throw std::runtime_error("Unable to get the output with variable values");
    }
    auto* data = output->m_sink->data.get();
    if(!data){
        return; /// The data is not allocated yet
    }
    auto* gpuArr = data->gpuArr.get();
    if(!gpuArr){
        return;
        //throw std::runtime_error("GPU data is not allocated");
    }
    FloatType* m_dev_root=gpuArr->devicePtr;
    for (size_t i = 0; i < size; ++i) {
        auto& val=m_variables[i].values();
        h_value_pointers_host[i]=val.data();
        h_value_pointers_dev[i]= m_dev_root+val.offset();
    }
    #else
    for (size_t i = 0; i < size; ++i) {
        auto& val=m_variables[i].values();
        h_value_pointers_host[i]=val.data();
        h_value_pointers_dev[i]=nullptr;
    }
    #endif
    allocateDevice();
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariables<FloatType,SizeType>::provideSignatureHost(SizeType &nvars, FloatType** &values){
    if(!m_tmanager){
        throw std::runtime_error("Unable to provide global GPU variables without TreeManager set");
    }
    nvars=h_value_pointers_host.size();
    values=h_value_pointers_host.data();
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariables<FloatType,SizeType>::provideSignatureDevice(SizeType &nvars, FloatType** &values){
    if(!m_tmanager){
        throw std::runtime_error("Unable to provide global GPU variables without TreeManager set");
    }
//    std::cout << __PRETTY_FUNCTION__ << "h_value_pointers_dev size= " <<  h_value_pointers_dev.size() << ", h_value_pointers_host size = " << h_value_pointers_host.size()  << " " << (void*)this <<std::endl <<std::endl;

    nvars=h_value_pointers_host.size();
    values=d_value_pointers_dev;
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariables<FloatType,SizeType>::deAllocateDevice(){
    if(d_value_pointers_dev){
#ifdef GNA_CUDA_SUPPORT
        cuwr_free<FloatType*>(d_value_pointers_dev);
#endif
        d_value_pointers_dev=nullptr;
    }
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariables<FloatType,SizeType>::allocateDevice(){
//    std::cout << "allocate dev " << __PRETTY_FUNCTION__ << " alloc size " << h_value_pointers_dev.size()  << " " << (void*)this << std::endl ;
#ifdef GNA_CUDA_SUPPORT
    //device_malloc(d_value_pointers_dev, h_value_pointers_dev.size());
    copyH2D_ALL(d_value_pointers_dev, h_value_pointers_dev.data(), h_value_pointers_dev.size());
#endif
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariables<FloatType,SizeType>::dump(const std::string& type){
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

template class TransformationTypes::GPUVariables<double,size_t>;
#ifdef PROVIDE_SINGLE_PRECISION
template class TransformationTypes::GPUVariables<float,size_t>;
#endif
