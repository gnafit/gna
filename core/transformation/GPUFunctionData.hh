#pragma once

#include <vector>
#include <iostream>
#include <Eigen/Dense>


#include "config_vars.h"
#ifdef GNA_CUDA_SUPPORT
#include "GpuBasics.hh"
#endif

namespace TransformationTypes{
    enum class GPUShape {
      Ndim = 0,  ///< Number of dimensions (position)
      Size,      ///< Size
      Nx,        ///< Size over first dimension (X)
      Ny,        ///< Size over second dimension (Y)
      Nz,        ///< Size over third dimension (Z)
    };

    template<typename FloatType,typename SizeType=unsigned int>
    class GPUFunctionData {
    public:
        GPUFunctionData(){ }
        ~GPUFunctionData(){ deAllocateDevice(); }

        template<typename DataContainer>
        void fillContainers(DataContainer& container);                             ///< Read the date from Source/Sink/Storage containers

        template<typename DataContainer>
        void fillContainersHost(DataContainer& container);                         ///< Read the date from Source/Sink/Storage containers

        template<typename DataContainer>
        void fillContainersDevice(DataContainer& container);                       ///< Read the date from gpu arrays of Source/Sink/Storage containers

        void allocateHost(size_t size);                                            ///< Reallocate memory (Host)
        void allocateDevice();                                                     ///< Reallocate memory (Device)
        void deAllocateDevice();                                                   ///< Deallocate memory (Device)

        void provideSignatureHost(SizeType &ndata, FloatType** &data, SizeType** &datashapes);    ///< Provide the pointers for the GPU arguments
        void provideSignatureDevice(SizeType &ndata, FloatType** &data, SizeType** &datashapes);  ///< Provide the pointers for the GPU arguments

        void dump(const std::string& type);

    private:
        std::vector<FloatType*> h_pointers;                    ///< Host Vector of pointers to data buffers (Host)
        std::vector<SizeType>   h_shapes;                      ///< Host Vector of shapes: ndim1 (2), size1 (dim1a*dim1b), dim1a, dim1b, ndim2(1), size2 (dim2), dim2, size3, ...
        std::vector<SizeType*>  h_shape_pointers_host;         ///< Host Vector of pointers to the relevant dimensions (Host)
        std::vector<SizeType>   h_offsets;                     ///< Host Vector of shape offsets (Host)

        std::vector<FloatType*> h_pointers_dev;                ///< Host Vector of pointers to data buffers (Device)
        std::vector<SizeType*>  h_shape_pointers_dev;          ///< Host Vector of pointers to the relevant dimensions (Dev)

        FloatType**             d_pointers_dev{nullptr};       ///< Device vector of pointers to data buffers (Device)
        SizeType*               d_shapes{nullptr};             ///< Device vector of shapes: ndim1 (2), size1 (dim1a*dim1b), dim1a, dim1b, ndim2(1), size2 (dim2), dim2, size3, ...
        SizeType**              d_shape_pointers_dev{nullptr}; ///< Device vector of pointers to the relevant dimensions (Device)
    };

    /**
     * @brief Reset the Host vectors and reserve space for N elements
     * @param size -- number of inputs/outputs/storages to allocate the memory for
     */
    template<typename FloatType,typename SizeType>
    void GPUFunctionData<FloatType,SizeType>::allocateHost(size_t size){
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

    /**
     * @brief Clean the memory allocated on the Device
     *
     * This method is also called in the destructor.
     */
    template<typename FloatType,typename SizeType>
    void GPUFunctionData<FloatType,SizeType>::deAllocateDevice(){
        size_t sh_size = h_shape_pointers_host.size();
        if(d_pointers_dev){
            for (size_t i =0; i < sh_size; i++) {
                cuwr_free<FloatType>(d_pointers_dev[i]);
            }
            cuwr_free<FloatType*>(d_pointers_dev);
        }
        if(d_shapes){
            cuwr_free<SizeType>(d_shapes);
        }
        if(d_shape_pointers_dev){
            for (size_t i =0; i < sh_size; i++) {
                cuwr_free<SizeType>(d_shape_pointers_dev[i]);
            }
            cuwr_free<SizeType*>(d_shape_pointers_dev);
        }
    }

    /**
     * @brief Allocate the memory on Device and fill with relevant data
     *
     * 1. Allocate the memory for the list of pointers to data (Device). Copy the pointers.
     * 2. Allocate the memory for the list of shape data (Device). Copy the shape data.
     * 3. Create host vector with pointers (Device) to the shape, relevant for each source/sink/storage.
     * 4. Allocate the memory for the list of pointers to shape data (Device). Copy the pointers.
     */
    template<typename FloatType,typename SizeType>
    void GPUFunctionData<FloatType,SizeType>::allocateDevice(){
        deAllocateDevice();

        copyH2D<FloatType*>(d_pointers_dev, h_pointers_dev.data(), h_pointers_dev.size());
        copyH2D<SizeType>(d_shapes, h_shapes.data(), h_shapes.size());

        size_t sh_size = h_shape_pointers_host.size();
        for (size_t i = 0; i< sh_size; i++) {
                copyH2D<SizeType>(h_shape_pointers_dev[i],h_shape_pointers_host[i], h_shapes[h_offsets[i]]);
        }
        copyH2D<SizeType*>(d_shape_pointers_dev, h_shape_pointers_dev.data(), h_shape_pointers_dev.size());
    }

    /**
     * @brief Fill the data from the sources/sinks/storages to the Device-friently list of pointers
     * @param container -- list of source/sink/storage instances
     */
    template<typename FloatType,typename SizeType>
    template<typename DataContainer>
    void GPUFunctionData<FloatType,SizeType>::fillContainers(DataContainer& container){
        fillContainersHost(container);
#ifdef GNA_CUDA_SUPPORT
        fillContainersDevice(container);
#endif
    }

    /**
     * @brief Walk over the list of sinks/sources/storages and fill the data types (Host)
     * @param container -- list of source/sink/storage instances
     */
    template<typename FloatType,typename SizeType>
    template<typename DataContainer>
    void GPUFunctionData<FloatType,SizeType>::fillContainersHost(DataContainer& container){
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

    template<typename FloatType,typename SizeType>
    template<typename DataContainer>
    void GPUFunctionData<FloatType,SizeType>::fillContainersDevice(DataContainer& container){
        for (size_t i = 0; i < container.size(); ++i) {
            if(container[i].materialized()){
                h_pointers_dev.push_back(container[i].gpuArr->devicePtr);
            }
            else{
                h_pointers_dev.push_back(nullptr);
            }
        }
        allocateDevice();
    }

    template<typename FloatType,typename SizeType>
    void GPUFunctionData<FloatType,SizeType>::provideSignatureHost(SizeType &ndata, FloatType** &data, SizeType** &datashapes){
        ndata     =h_pointers.size();
        data      =h_pointers.data();
        datashapes=h_shape_pointers_host.data();
    }

    template<typename FloatType,typename SizeType>
    void GPUFunctionData<FloatType,SizeType>::provideSignatureDevice(SizeType &ndata, FloatType** &data, SizeType** &datashapes){
        ndata     =h_pointers_dev.size();
        data      =d_pointers_dev;
        datashapes=d_shape_pointers_dev;
    }

    template<typename FloatType,typename SizeType>
    void GPUFunctionData<FloatType,SizeType>::dump(const std::string& type){
        size_t      ndata     =h_pointers.size();
        FloatType** datas     =h_pointers.data();
        SizeType**  datashapes=h_shape_pointers_host.data();

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
}
