#include <gsl/gsl_integration.h>

#include <algorithm>

#include <Eigen/Dense>

#include "GaussLegendre.hh"

using namespace Eigen;

void GaussLegendre::init() {
  size_t npoints = 0;
  for (size_t n: m_orders) {
    npoints += n;
  }
  m_points.resize(npoints);
  m_weights.resize(npoints);
  size_t k = 0;
  for (size_t i = 0; i < m_orders.size(); ++i) {
    size_t n = m_orders[i];
    auto *t = gsl_integration_glfixed_table_alloc(n);

    double a = m_edges[i];
    double b = m_edges[i+1];

    for (size_t j = 0; j < t->n; ++j) {
      gsl_integration_glfixed_point(a, b, j, &m_points[k], &m_weights[k], t);
      k++;
    }
    gsl_integration_glfixed_table_free(t);
  }
  transformation_(this, "points")
    .output("x")
    .types([](GaussLegendre *obj, Atypes, Rtypes rets) {
        rets[0] = DataType().points().shape(obj->m_points.size());
      })
    .func([](GaussLegendre *obj, Args, Rets rets) {
        rets[0].x = obj->m_points;
      })
    ;
}

GaussLegendreHist::GaussLegendreHist(const GaussLegendre *base)
  : m_base(base)
{
  transformation_(this, "hist")
    .input("f")
    .output("hist")
    .types(Atypes::ifSame, [](GaussLegendreHist *obj, Atypes, Rtypes rets) {
        rets[0] = DataType().hist().bins(obj->m_base->m_orders.size()).edges(obj->m_base->m_edges);
      })
    .func([](GaussLegendreHist *obj, Args args, Rets rets) {
        ArrayXd prod = args[0].x*obj->m_base->m_weights;
        auto *data = prod.data();
        const auto &orders = obj->m_base->m_orders;
        for (size_t i = 0; i < orders.size(); ++i) {
          size_t n = orders[i];
          rets[0].x(i) = std::accumulate(data, data+n, 0.0);
          data += n;
        }
      })
    ;  
}
