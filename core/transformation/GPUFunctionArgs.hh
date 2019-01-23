#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include "TransformationEntry.hh"

namespace TransformationTypes{
    enum class GPUShape {
      Ndim = 0,  ///< Number of dimensions (position)
      Size,      ///< Size
      Nx,        ///< Size over first dimension (X)
      Ny,        ///< Size over second dimension (Y)
      Nz,        ///< Size over third dimension (Z)
    };

    template<typename FloatType>
    struct GPUFunctionData {
        using SizeType = unsigned int;
        GPUFunctionData(){ }
        ~GPUFunctionData(){ deAllocateDevice(); }

        template<typename DataContainer>
        void fillContainers(DataContainer& container);                             ///< Read the date from Source/Sink/Storage containers

        template<typename DataContainer>
        void fillContainersHost(DataContainer& container);                         ///< Read the date from Source/Sink/Storage containers

        void fillContainersDevice();                                               ///< Read the date from Source/Sink/Storage containers

        void allocateHost(size_t size);                                            ///< Reallocate memory (Host)
        void allocateDevice();                                                     ///< Reallocate memory (Device)
        void deAllocateDevice();                                                   ///< Deallocate memory (Device)

        void provideInfoHost(SizeType &ndata, FloatType** &data, SizeType** &datashapes);    ///< Provide the pointers for the GPU arguments
        void provideInfoDevice(SizeType &ndata, FloatType** &data, SizeType** &datashapes);  ///< Provide the pointers for the GPU arguments

        void dump(const std::string& type);

        std::vector<FloatType*> h_pointers;              ///< Host Vector of pointers to data buffers (Host)
        std::vector<SizeType>   h_shapes;                ///< Host Vector of shapes: ndim1 (2), size1 (dim1a*dim1b), dim1a, dim1b, ndim2(1), size2 (dim2), dim2, size3, ...
        std::vector<SizeType*>  h_shape_pointers_host;   ///< Host Vector of pointers to the relevant dimensions (Host)
        std::vector<SizeType>   h_offsets;               ///< Host Vector of shape offsets (Host)

        std::vector<FloatType*> h_pointers_dev;          ///< Host Vector of pointers to data buffers (Device)
        std::vector<SizeType*>  h_shape_pointers_dev;          ///< Host Vector of pointers to the relevant dimensions (Dev)

        FloatType**             d_pointers_dev{nullptr};       ///< Device vector of pointers to data buffers (Device)
        SizeType*               d_shapes{nullptr};             ///< Device vector of shapes: ndim1 (2), size1 (dim1a*dim1b), dim1a, dim1b, ndim2(1), size2 (dim2), dim2, size3, ...
        SizeType**              d_shape_pointers_dev{nullptr}; ///< Device vector of pointers to the relevant dimensions (Device)
    };

    template<typename FloatType>
    struct GPUFunctionArgsT {
    public:
        using SizeType = unsigned int;
        GPUFunctionArgsT(Entry* entry) : m_entry(entry){

        }

        ~GPUFunctionArgsT(){

        }

        void updateTypes();
        void dump();

        SizeType    npars{0u};       // number of parameters
        FloatType **pars{nullptr};   // list of pointers to parameter values
        SizeType    nargs{0u};       // number of args
        FloatType **args{nullptr};   // list of pointers to args
        SizeType  **argshapes{0u};   //
        SizeType    nrets{0u};       // number of rets
        FloatType **rets{nullptr};   // list of pointers to rets
        SizeType  **retshapes{0u};   //
        SizeType    nints{0u};       // number of ints
        FloatType **ints{nullptr};   // list of pointers to ints
        SizeType  **intshapes{0u};   //

    private:
        Entry* m_entry;

        std::vector<FloatType*> m_pars;
        GPUFunctionData<FloatType> m_args;
        GPUFunctionData<FloatType> m_rets;
        GPUFunctionData<FloatType> m_ints;
    };

    template<typename FloatType>
    void GPUFunctionArgsT<FloatType>::updateTypes(){
        m_args.fillContainers(m_entry->sources);
        m_args.provideInfoHost(nargs, args, argshapes);

        m_rets.fillContainers(m_entry->sinks);
        m_rets.provideInfoHost(nrets, rets, retshapes);

        m_ints.fillContainers(m_entry->storages);
        m_ints.provideInfoHost(nints, ints, intshapes);
    }

    template<typename FloatType>
    void GPUFunctionArgsT<FloatType>::dump(){
        printf("Dumping GPU args state\n");

        printf("    ");
        m_args.dump("sources");
        printf("\n");

        printf("    ");
        m_rets.dump("sinks");
        printf("\n");

        printf("    ");
        m_ints.dump("storages");
        printf("\n");

    }

    template<typename FloatType>
    void GPUFunctionData<FloatType>::allocateHost(size_t size){
        h_pointers.clear();
        h_pointers.reserve(size);
        h_shape_pointers_host.clear();
        h_shape_pointers_host.reserve(size);
        h_shapes.clear();
        h_shapes.reserve(4*size);

        h_pointers_dev.clear();
        h_pointers_dev.reserve(size);
        h_shape_pointers_dev.clear();
        h_shape_pointers_dev.reserve(size);
    }

    template<typename FloatType>
    void GPUFunctionData<FloatType>::deAllocateDevice(){
        if(d_pointers_dev){
            // TODO:
            //size of h_pointers_dev.size()
            d_pointers_dev=nullptr;
        }
        if(d_shapes){
            // TODO:
            //size of h_shapes.size()
            d_shapes=nullptr;
        }
        if(d_shape_pointers_dev){
            // TODO:
            //size of h_pointers_dev.size()
            d_shape_pointers_dev=nullptr;
        }
    }

    template<typename FloatType>
    void GPUFunctionData<FloatType>::allocateDevice(){
        deAllocateDevice();

        // TODO:
        // h_pointers_dev -> d_pointers
        // h_shapes       -> d_shapes

        // Calculate the pointers to shape of each Data based on offsets
        //auto* initial=d_shapes.data();
        //for (size_t i = 0; i < h_offsets.size(); ++i) {
            //h_shape_pointers_dev[i] = next(initial, h_offsets[i]);
        //}

        // TODO:
        // h_shape_pointers_dev -> d_shape_pointers_dev
    }

    template<typename FloatType>
    template<typename DataContainer>
    void GPUFunctionData<FloatType>::fillContainers(DataContainer& container){
        fillContainersHost(container);
        fillContainersDevice();
    }

    template<typename FloatType>
    void GPUFunctionData<FloatType>::fillContainersDevice(){
    }

    template<typename FloatType>
    template<typename DataContainer>
    void GPUFunctionData<FloatType>::fillContainersHost(DataContainer& container){
        allocateHost(container.size());

        for (size_t i = 0; i < container.size(); ++i) {
            if(container[i].materialized()){
                auto* data=container[i].getData();
                h_pointers.push_back(data->buffer);

                auto& shape=data->type.shape;
                auto offset=h_shapes.size();
                h_offsets.push_back(offset);
                h_shapes.push_back(shape.size());
                h_shapes.push_back(data->type.size());
                h_shapes.insert(h_shapes.end(), shape.begin(), shape.end());

                h_shape_pointers_host.push_back(&h_shapes[offset]);
            }
            else{
                h_pointers.push_back(nullptr);
                auto offset=h_shapes.size();
                h_offsets.push_back(offset);
                h_shapes.push_back(0u);
                h_shapes.push_back(0u);
                h_shape_pointers_host.push_back(&h_shapes[offset]);
            }
        }
    }

    template<typename FloatType>
    void GPUFunctionData<FloatType>::provideInfoHost(SizeType &ndata, FloatType** &data, SizeType** &datashapes){
        ndata     =h_pointers.size();
        data      =h_pointers.data();
        datashapes=h_shape_pointers_host.data();
    }

    template<typename FloatType>
    void GPUFunctionData<FloatType>::provideInfoDevice(SizeType &ndata, FloatType** &data, SizeType** &datashapes){
        ndata     =h_pointers_dev.size();
        data      =d_pointers_dev;
        datashapes=d_shape_pointers_dev;
    }

    template<typename FloatType>
    void GPUFunctionData<FloatType>::dump(const std::string& type){
        size_t      ndata     =h_pointers.size();
        FloatType** datas     =h_pointers.data();
        SizeType**       datashapes=h_shape_pointers_host.data();

        printf("Dumping GPUFunctionData of size %zu", ndata);
        if(type.size()){
            printf(" (%s)", type.c_str());
        }
        printf("\n");
        for (size_t i = 0; i < ndata; ++i) {
            FloatType* data=datas[i];
            SizeType* shapedef=datashapes[i];
            SizeType  ndim=shapedef[(size_t)GPUShape::Ndim];
            SizeType  size=shapedef[(size_t)GPUShape::Size];
            SizeType* shape=std::next(datashapes[i], (size_t)GPUShape::Nx);
            printf("Data %zu of size %zu, ndim %zu, ptr %p", i, (size_t)size, (size_t)ndim, (void*)data);
            if(ndim){
                printf(", shape %zu", (size_t)shape[0]);
                for (SizeType j = 1; j<ndim; ++j) {
                    printf("x%zu", (size_t)shape[j]);
                }
                printf("\n");
                if(ndim==2){
                    Eigen::Map<Eigen::Array<FloatType,Eigen::Dynamic,Eigen::Dynamic>> view(data, shape[0], shape[1]);
                    std::cout<<view<<std::endl;
                }
                else{
                    Eigen::Map<Eigen::Array<FloatType,Eigen::Dynamic,1>> view(data, size);
                    std::cout<<view<<std::endl;
                }
            }
            printf("\n");
        }
    }

    template class GPUFunctionData<double>;
    template class GPUFunctionArgsT<double>;

    #ifdef PROVIDE_SINGLE_PRECISION
        template class GPUFunctionData<float>;
        template class GPUFunctionArgsT<float>;
    #endif
    using GPUFunctionArgs = GPUFunctionArgsT<double>;
}
