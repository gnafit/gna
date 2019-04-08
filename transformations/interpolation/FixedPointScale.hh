#pragma once

#include <string>
#include "InSegment.hh"

class FixedPointScale: public InSegment,
                       public TransformationBind<FixedPointScale> {
public:
  using TransformationBind<FixedPointScale>::transformation_;

  FixedPointScale();                                                                                                      ///< Constructor.
  FixedPointScale(SingleOutput& x, SingleOutput& fixedpoint);                                                             ///< Constructor.
  FixedPointScale(SingleOutput& x, SingleOutput& y, SingleOutput& fixedpoint);                                            ///< Constructor.

  TransformationDescriptor add_transformation(const std::string& name="");
  void bind_transformations();
  void bind_inputs();
  void set(SingleOutput& x, SingleOutput& fixedpoint);

  OutputDescriptor scale(SingleOutput& x, SingleOutput& y, SingleOutput& fixedpoint);                          ///< Initialize transformations by connecting `x`, `y` and `newy` outputs.

protected:
  void do_scale(FunctionArgs& fargs);                                                                    ///< Do the scaling.
  void check(TypesFunctionArgs& fargs);                                                                    ///< Do the scaling.
};
