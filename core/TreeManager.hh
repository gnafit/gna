#pragma once

#include <boost/noncopyable.hpp>

namespace GNA{
    template<typename FloatType>
    class TreeManager : public boost::noncopyable
    {
    protected:
        using TreeManagerType = TreeManager<FloatType>;

    public:
        TreeManager()  {
        }
        virtual ~TreeManager(){

        }

    protected:
        const size_t m_max_size=1000;
        /* data */
    };
}
