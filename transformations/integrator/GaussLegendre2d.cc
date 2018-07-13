#include <gsl/gsl_integration.h>

#include <algorithm>

#include <Eigen/Dense>

#include "GaussLegendre2d.hh"
#include "TypesFunctions.hh"

using namespace Eigen;

void GaussLegendre2d::init() {
  size_t npoints = 0;
  for (size_t n: m_xorders) {
    npoints += n;
  }
  m_xpoints.resize(npoints);
  m_xweights.resize(npoints);
  size_t k = 0;
  for (size_t i = 0; i < m_xorders.size(); ++i) {
    size_t n = m_xorders[i];
    auto *t = gsl_integration_glfixed_table_alloc(n);

    double a = m_xedges[i];
    double b = m_xedges[i+1];

    for (size_t j = 0; j < t->n; ++j) {
      gsl_integration_glfixed_point(a, b, j, &m_xpoints[k], &m_xweights[k], t);
      k++;
    }
    gsl_integration_glfixed_table_free(t);
  }

  auto *t = gsl_integration_glfixed_table_alloc(m_yorder);
  m_ypoints.resize(t->n);
  m_yweights.resize(t->n);
  for (size_t i = 0; i < t->n; ++i) {
    gsl_integration_glfixed_point(m_ymin, m_ymax, i,
                                  &m_ypoints[i], &m_yweights[i], t);
  }
  gsl_integration_glfixed_table_free(t);

  transformation_("points")
    .output("x")
    .output("y")
    .output("xedges")
    .types([](GaussLegendre2d *obj, Atypes, Rtypes rets) {
        rets[0] = DataType().points().shape(obj->m_xpoints.size());
        rets[1] = DataType().points().shape(obj->m_ypoints.size());
        rets[2] = DataType().points().shape(obj->m_xedges.size());
      })
    .func([](GaussLegendre2d *obj, FunctionArgs fargs) {
        auto& rets=fargs.rets;
        rets[0].x = Eigen::Map<const Eigen::ArrayXd>(&obj->m_xpoints[0], obj->m_xpoints.size());
        rets[1].x = Eigen::Map<const Eigen::ArrayXd>(&obj->m_ypoints[0], obj->m_ypoints.size());
        rets[2].x = Eigen::Map<const Eigen::ArrayXd>(&obj->m_xedges[0], obj->m_xedges.size());
        rets.freeze();
      })
    .finalize()
    ;
}

GaussLegendre2dHist::GaussLegendre2dHist(const GaussLegendre2d *base)
  : m_base(base)
{
  transformation_("hist")
    .input("f")
    .output("hist")
    .types(TypesFunctions::ifSame, [](GaussLegendre2dHist *obj, Atypes, Rtypes rets) {
        rets[0] = DataType().hist().bins(obj->m_base->m_xorders.size()).edges(obj->m_base->m_xedges);
      })
    .func([](GaussLegendre2dHist *obj, FunctionArgs fargs) {
        auto& args=fargs.args;
        auto& rets=fargs.rets;
        size_t shape[2];
        shape[0] = obj->m_base->m_xpoints.size();
        shape[1] = obj->m_base->m_ypoints.size();
        for (size_t k = 0; k < rets.size(); ++k) {
          Map<const ArrayXXd, Aligned> pts(args[k].x.data(), shape[0], shape[1]);
          auto &xw = obj->m_base->m_xweights, &yw = obj->m_base->m_yweights;
          ArrayXd prod = (pts.rowwise()*yw.transpose()).rowwise().sum()*xw;
          auto *data = prod.data();
          for (size_t i = 0; i < obj->m_base->m_xorders.size(); ++i) {
            size_t n = obj->m_base->m_xorders[i];
            rets[k].x(i) = std::accumulate(data, data+n, 0.0);
            data += n;
          }
        }
      })
    ;
}

