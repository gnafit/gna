#pragma once

#include <iostream>
#include <cmath>
#include <string>

#include "GNAObject.hh"
#include "TypesFunctions.hh"
#include <Eigen/Dense>


class GeoNeutrinoFluxNormed: public GNASingleObject,
                public TransformationBind<GeoNeutrinoFluxNormed> {
public:
  GeoNeutrinoFluxNormed(double livetime_years): m_livetime_years(livetime_years) {
    variable_(&m_fluxnorm, "FluxNorm");
    transformation_("flux_norm")
      .input("flux")
      .output("normed_flux")
      .types(TypesFunctions::ifSame, TypesFunctions::pass<0>)
      .func(&GeoNeutrinoFluxNormed::CalcNorm);
  }
protected:
  void CalcNorm(FunctionArgs& fargs) {
      auto& args=fargs.args;
      const double* events = args[0].x.data();
      const size_t insize = args[0].type.size();
      size_t first_non_nan_idx{0};
      // find first not nan assuming all other values are not NaN!
      for (; first_non_nan_idx < insize; ++first_non_nan_idx) {
        if (std::isnan(events[first_non_nan_idx])) {
            ++first_non_nan_idx;
        }
        else {
            break;
        }
      }
      const double total_events = std::accumulate(events + first_non_nan_idx, events + insize, 0.);

      fargs.rets[0].x = (m_fluxnorm/total_events) * m_livetime_years * fargs.args[0].x;
  }

  variable<double> m_fluxnorm;

  double m_livetime_years;
};
