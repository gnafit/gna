#pragma once

#include <boost/noncopyable.hpp>
#include <memory>
#include "variable.hh"

template<typename FloatType>
class arrayviewAllocator;

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
    public:
        using allocatorType = arrayviewAllocator<FloatType>;
        using allocatorPtr = std::unique_ptr<allocatorType>;
        using variableType = variable<FloatType>;

        TreeManager(size_t allocatepars=0u);
        virtual ~TreeManager();

        void setVariables(const std::vector<variableType>& vars);

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
    protected:
        static void setCurrent(TreeManagerType* tmanager) noexcept { TreeManagerType::s_current_manager=tmanager; }

        void resetAllocator() const;
        void setAllocator() const;

        static TreeManagerType* s_current_manager;
        allocatorPtr m_allocator=nullptr;
        VarArrayPtr m_vararray=nullptr;
    };

    template<typename T> TreeManager<T>* TreeManager<T>::s_current_manager=nullptr;
}
