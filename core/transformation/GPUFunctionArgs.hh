#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include "TransformationEntry.hh"

namespace TransformationTypes{
    template<typename FloatType>
    struct GPUFunctionData {
        GPUFunctionData(){

        }

        template<typename DataContainer>
        void fillContainers(DataContainer& container);

        void provideInfo();

        void dump(const std::string& type){
            size_t      ndata     =pointers.size();
            FloatType** datas     =pointers.data();
            size_t**    datashapes=shape_pointers.data();

            auto margin=std::cout.tellp();
            printf("Dumping GPUFunctionData of size %zu", ndata);
            if(type.size()){
                printf(" (%s)", type.c_str());
            }
            printf("\n");
            for (size_t i = 0; i < ndata; ++i) {
                std::cout.seekp(margin);

                FloatType* data=datas[i];
                size_t* shapedef=datashapes[i];
                size_t  ndim=shapedef[0];
                size_t  size=shapedef[1];
                size_t* shape=std::next(datashapes[i], 2);
                printf("Data %zu of size %zu, ndim %zu, shape %zu", i, size, ndim, shape[0]);
                for (size_t j = 1; j<ndim; ++j) {
                    printf("x%zu", shape[j]);
                }
                std::cout<<std::endl;

                std::cout.seekp(margin);
                printf("  ");
                if(ndim==2){
                    Eigen::Map<Eigen::Array<FloatType,Eigen::Dynamic,Eigen::Dynamic>> view(data, shape[0], shape[1]);
                    std::cout<<view;
                }
                else{
                    Eigen::Map<Eigen::Array<FloatType,Eigen::Dynamic,1>> view(data, size);
                    std::cout<<view;
                }
                std::cout<<std::endl;
            }
        }

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

        void evaluateTypes();
        void dump(){
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

    private:
        Entry* m_entry;

        std::vector<FloatType*> m_pars;
        GPUFunctionData<FloatType> m_args;
        GPUFunctionData<FloatType> m_rets;
        GPUFunctionData<FloatType> m_ints;
    };

    template<typename FloatType>
    void GPUFunctionArgsT<FloatType>::evaluateTypes(){
        m_args.fillContainers(m_entry->sources);
        m_rets.fillContainers(m_entry->sinks);
        m_ints.fillContainers(m_entry->storages);
    }

    //template<typename FloatType>
    //void GPUFunctionArgsT<FloatType>::dump(){
    //}

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
            auto* data=container[i].getData();
            pointers.push_back(data->buffer);

            auto& shape=data->type.shape;
            auto offset=shapes.size();
            shapes.push_back(shape.size());
            shapes.push_back(data->type.size());
            shapes.insert(shapes.end(), shape.begin(), shape.end());

            shape_pointers.push_back(&shapes[offset]);
        }
    }

    template<typename FloatType>
    void GPUFunctionData<FloatType>::provideInfo(){
        //size_t      ndata     =pointers.size();
        //FloatType** data      =pointers.data();
        //size_t**    datashapes=shape_pointers.data();
    }

    //template<typename FloatType>
    //void GPUFunctionData<FloatType>::dump(const std::string& type){
    //}

    template class GPUFunctionData<double>;
    template class GPUFunctionArgsT<double>;
    using GPUFunctionArgs = GPUFunctionArgsT<double>;
}
