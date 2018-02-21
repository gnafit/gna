#pragma once

#include <vector>
#include <limits>
#include <iostream>

#include "GNAObject.hh"

enum class ReturnOnFail: int {UseNaN, UseZero};

class LinearInterpolator: public GNASingleObject,
                          public TransformationBind<LinearInterpolator> {
public:
  LinearInterpolator(int size, const double *xs, const double *ys, std::string return_on_fail = "")
    : m_xs(xs, xs+size), m_ys(ys, ys+size)
  {

    indexBins();
    transformation_("f")
      .input("x")
      .output("y")
      .types(Atypes::pass<0>)
      .func(&LinearInterpolator::interpolate) ;

   if (return_on_fail == "use_zero") {
       m_status_on_fail = ReturnOnFail::UseZero;
   }
   else {
       m_status_on_fail = ReturnOnFail::UseNaN;
   }

  }
protected:
  void indexBins();
  void interpolate(Args args, Rets rets);

  std::vector<double> m_xs;
  std::vector<double> m_ys;
  ReturnOnFail m_status_on_fail;
  std::vector<size_t> m_index;
  double m_minbinsize;
};
