#pragma once

#include <boost/noncopyable.hpp>
#include <memory>

template<typename FloatType>
class arrayviewAllocator;

namespace GNA{
    template<typename FloatType>
    class TreeManager: public boost::noncopyable
    {
    protected:
        using TreeManagerType = TreeManager<FloatType>;
    public:
        using allocatorType = arrayviewAllocator<FloatType>;
        using allocatorPtr = std::unique_ptr<allocatorType>;

        TreeManager(size_t allocatepars=0u);
        virtual ~TreeManager();

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
    };

    template<typename T> TreeManager<T>* TreeManager<T>::s_current_manager=nullptr;
}
