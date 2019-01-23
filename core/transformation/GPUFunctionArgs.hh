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
        GPUFunctionData(){ }

        template<typename DataContainer>
        void fillContainers(DataContainer& container);                              ///< Read the date from Source/Sink/Storage containers
        void provideInfoCPU(size_t &ndata, FloatType** &data, size_t** &datashapes); ///< Provide the pointers for the GPU arguments
        void provideInfoGPU(size_t &ndata, FloatType** &data, size_t** &datashapes); ///< Provide the pointers for the GPU arguments

        void dump(const std::string& type);

        std::vector<FloatType*> pointers;        ///< Vector of pointers to data buffers
        std::vector<size_t>     shapes;          ///< Vector of shapes: ndim1 (2), size1 (dim1a*dim1b), dim1a, dim1b, ndim2(1), size2 (dim2), dim2, size3, ...
        std::vector<size_t*>    shape_pointers;  ///< Vector of pointers to the relevant dimensions
    };

    template<typename FloatType>
    struct GPUFunctionArgsT {
    public:
        GPUFunctionArgsT(Entry* entry) : m_entry(entry){

        }

        ~GPUFunctionArgsT(){

        }

        void updateTypes();
        void dump();

        size_t      npars{0u};       // number of parameters
        FloatType **pars{nullptr};   // list of pointers to parameter values
        size_t      nargs{0u};       // number of args
        FloatType **args{nullptr};   // list of pointers to args
        size_t    **argshapes{0u};   //
        size_t      nrets{0u};       // number of rets
        FloatType **rets{nullptr};   // list of pointers to rets
        size_t    **retshapes{0u};   //
        size_t      nints{0u};       // number of ints
        FloatType **ints{nullptr};   // list of pointers to ints
        size_t    **intshapes{0u};   //

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
        m_args.provideInfoCPU(nargs, args, argshapes);

        m_rets.fillContainers(m_entry->sinks);
        m_rets.provideInfoCPU(nrets, rets, retshapes);

        m_ints.fillContainers(m_entry->storages);
        m_ints.provideInfoCPU(nints, ints, intshapes);
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
    template<typename DataContainer>
    void GPUFunctionData<FloatType>::fillContainers(DataContainer& container){
        // Allocate memory
        pointers.clear();
        pointers.reserve(container.size());
        shape_pointers.clear();
        shape_pointers.reserve(container.size());
        shapes.clear();
        shapes.reserve(4*container.size());

        for (size_t i = 0; i < container.size(); ++i) {
            if(container[i].materialized()){
                auto* data=container[i].getData();
                pointers.push_back(data->buffer);

                auto& shape=data->type.shape;
                auto offset=shapes.size();
                shapes.push_back(shape.size());
                shapes.push_back(data->type.size());
                shapes.insert(shapes.end(), shape.begin(), shape.end());

                shape_pointers.push_back(&shapes[offset]);
            }
            else{
                pointers.push_back(nullptr);
                auto offset=shapes.size();
                shapes.push_back(0u);
                shapes.push_back(0u);
                shape_pointers.push_back(&shapes[offset]);
            }
        }
    }

    template<typename FloatType>
    void GPUFunctionData<FloatType>::provideInfoCPU(size_t &ndata, FloatType** &data, size_t** &datashapes){
        ndata     =pointers.size();
        data      =pointers.data();
        datashapes=shape_pointers.data();
    }

    template<typename FloatType>
    void GPUFunctionData<FloatType>::dump(const std::string& type){
        size_t      ndata     =pointers.size();
        FloatType** datas     =pointers.data();
        size_t**    datashapes=shape_pointers.data();

        printf("Dumping GPUFunctionData of size %zu", ndata);
        if(type.size()){
            printf(" (%s)", type.c_str());
        }
        printf("\n");
        for (size_t i = 0; i < ndata; ++i) {
            FloatType* data=datas[i];
            size_t* shapedef=datashapes[i];
            size_t  ndim=shapedef[(int)GPUShape::Ndim];
            size_t  size=shapedef[(int)GPUShape::Size];
            size_t* shape=std::next(datashapes[i], (int)GPUShape::Nx);
            printf("Data %zu of size %zu, ndim %zu, ptr %p", i, size, ndim, (void*)data);
            if(ndim){
                printf(", shape %zu", shape[0]);
                for (size_t j = 1; j<ndim; ++j) {
                    printf("x%zu", shape[j]);
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
