#pragma once

#include "InSegment.hh"
#include <string>
#include <unsupported/Eigen/Splines>


class SplineFunction;

class InterpSpline: public InSegment,
                    public TransformationBind<InterpSpline> {
    public:
        using TransformationBind<InterpSpline>::transformation_;
        InterpSpline();                                                                                                ///< Constructor.
        InterpSpline(SingleOutput& x, SingleOutput& newx);                                                             ///< Constructor.
        InterpSpline(SingleOutput& x, SingleOutput& y, SingleOutput& newx);                                            ///< Constructor.

        TransformationDescriptor add_transformation(const std::string& name="");
        void bind_transformations();
        void bind_inputs();
        void set(SingleOutput& x, SingleOutput& newx);

        OutputDescriptor interpolate(SingleOutput& x, SingleOutput& y, SingleOutput& newx);                        ///< Initialize transformations by connecting `x`, `y` and `newy` outputs.
        void do_interpolate(FunctionArgs& fargs);


    private:
        // Spline of one-dimensional "points."
        Eigen::Spline<double, 1> spline_;
};
