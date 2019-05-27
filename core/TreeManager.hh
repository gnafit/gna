#pragma once

#include <boost/noncopyable.hpp>
#include <memory>
#include <set>
#include "config_vars.h"
#include "variable.hh"

template<typename FloatType>
class arrayviewAllocator;

template<typename SourceFloatType, typename SinkFloatType>
class OutputDescriptorT;

template<typename SourceFloatType, typename SinkFloatType>
class TransformationDescriptorT;

namespace TransformationTypes{
  template<typename SourceFloatType, typename SinkFloatType>
  class EntryT;
}

namespace GNA{
    namespace GNAObjectTemplates{
        template<typename FloatType>
        class VarArrayPreallocatedT;
    }

    template<typename FloatType>
    class TreeManager: public boost::noncopyable
    {
    protected:
        using TreeManagerType = TreeManager<FloatType>;
        using VarArrayType = GNAObjectTemplates::VarArrayPreallocatedT<FloatType>;
        using VarArrayPtr = std::unique_ptr<VarArrayType>;
        using OutputDescriptorType = OutputDescriptorT<FloatType,FloatType>;
        using OutputDescriptorPtr = std::unique_ptr<OutputDescriptorType>;
        using TransformationDescriptorType = TransformationDescriptorT<FloatType,FloatType>;
        using TransformationDescriptorPtr = std::unique_ptr<TransformationDescriptorType>;
    public:
        using allocatorType = arrayviewAllocator<FloatType>;
        using allocatorPtr = std::unique_ptr<allocatorType>;
        using variableType = variable<FloatType>;
        using TransformationType = TransformationTypes::EntryT<FloatType,FloatType>;

        TreeManager(size_t allocatepars=0u);
        virtual ~TreeManager();

        void setVariables(const std::vector<variableType>& vars);
        bool hasVariable(const variable<void>& variable);
        bool consistentVariable(const variable<FloatType>& variable);

        void registerTransformation(TransformationType* entry){ m_transformations.insert(entry); }
        bool hasTransformation(TransformationType* entry) { return m_transformations.find(entry)!=m_transformations.end(); }

        void update();

        static TreeManagerType* current() noexcept { return TreeManagerType::s_current_manager; }
        static void resetCurrent() noexcept {
            if(!TreeManagerType::s_current_manager){
                return;
            }
            TreeManagerType::s_current_manager->resetAllocator();
            TreeManagerType::s_current_manager=nullptr;
        }
        void makeCurrent();

        allocatorType* getAllocator() { return m_allocator.get(); }
        VarArrayType* getVarArray() { return m_vararray.get(); }
        OutputDescriptorType* getOutput() { return m_output.get(); }
    protected:
        static void setCurrent(TreeManagerType* tmanager) noexcept { TreeManagerType::s_current_manager=tmanager; }

        void resetAllocator() const;
        void setAllocator() const;

        static TreeManagerType* s_current_manager;
        allocatorPtr m_allocator=nullptr;
        VarArrayPtr m_vararray=nullptr;
        OutputDescriptorPtr m_output=nullptr;
        TransformationDescriptorPtr m_transformation=nullptr;

        std::set<TransformationType*> m_transformations;
    };

    template<typename T> TreeManager<T>* TreeManager<T>::s_current_manager=nullptr;
}
