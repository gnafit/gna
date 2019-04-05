#pragma once
#include "GNAObject.hh"
#include "TransformationBind.hh"
#include <boost/optional.hpp>

class PartialSum: public GNAObject,
                 public TransformationBind<PartialSum> {
    public:
        using TransformationBind<PartialSum>::transformation_;
        PartialSum();
        PartialSum(double initial_value);
        PartialSum(double initial_value, double threshold);
        PartialSum(double threshold, bool);

    private:
        void init();
        void calc(FunctionArgs fargs);
        void findIdx(TypesFunctionArgs targs);
        void make_Points(TypesFunctionArgs targs);

        template<typename InputIterator>
        void findStartingPoint(InputIterator start, InputIterator end);

        boost::optional<double> m_threshold;
        boost::optional<double> m_initial_value;
        size_t m_idx = 0;
};
