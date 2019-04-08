#pragma once
#include "GNAObject.hh"
#include "fragile.hh"
#include <boost/optional.hpp>

class PointsToHist: public GNAObject,
                    public TransformationBind<PointsToHist> {
    public:
        using TransformationBind<PointsToHist>::transformation_;
        PointsToHist(SingleOutput& points, double left_edge);
        PointsToHist(SingleOutput& points);

    private:
        void init(SingleOutput& points);
        boost::optional<double> m_left_edge;
        std::vector<double> m_edges;
        fragile m_fragile;
};
