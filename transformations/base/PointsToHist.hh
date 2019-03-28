#pragma once
#include "GNAObject.hh"


class PointsToHist: public GNAObject,
                    public TransformationBind<PointsToHist> {
    public:
        using TransformationBind<PointsToHist>::transformation_;
        PointsToHist(SingleOutput& points, double left_most_bin = 0.);
};
