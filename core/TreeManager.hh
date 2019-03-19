#pragma once

#include <boost/noncopyable.hpp>

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

        TreeManager()  { }
        virtual ~TreeManager(){ }

        void update(){}

        static TreeManagerType* current() noexcept { return TreeManagerType::s_current_manager; }
        static void setCurrent(TreeManagerType* tmanager) noexcept { TreeManagerType::s_current_manager=tmanager; }

    protected:
        void resetAllocator() const;
        void setAllocator() const;

        static TreeManagerType* s_current_manager;
        allocatorType* m_allocator=nullptr;
    };

    template<typename T> TreeManager<T>* TreeManager<T>::s_current_manager=nullptr;
}
