#include <gsl/gsl_integration.h>

#include <algorithm>

#include <Eigen/Dense>

#include "GaussLegendre.hh"
#include "TypesFunctions.hh"

using namespace Eigen;

void GaussLegendre::init() {
  size_t npoints = 0;
  for (size_t n: m_orders) {
    npoints += n;
  }
  m_points.resize(npoints);
  m_weights.resize(npoints);
  //k stands for a bin number
  size_t k = 0;

  //simply allocates the weights and abscissae table of Gauss-Legendre n-point
  //integration rule
  //https://en.wikipedia.org/wiki/Gaussian_quadrature
  for (size_t i = 0; i < m_orders.size(); ++i) {
    size_t n = m_orders[i];
    auto *t = gsl_integration_glfixed_table_alloc(n);

    double a = m_edges[i];
    double b = m_edges[i+1];

    //actually provides the absc and weight. for interval [a,b] and store it
    //to m_points and m_weights,
    for (size_t j = 0; j < t->n; ++j) {
      gsl_integration_glfixed_point(a, b, j, &m_points[k], &m_weights[k], t);
      k++;
    }
    gsl_integration_glfixed_table_free(t);
  }
  //return only points, WTF?
  transformation_("points")
    .output("x")
    .output("xedges")
    .types([](GaussLegendre *obj, TypesFunctionArgs fargs) {
        auto& rets=fargs.rets;
        rets[0] = DataType().points().shape(obj->m_points.size());
        rets[1] = DataType().points().shape(obj->m_edges.size());
      })
    .func([](GaussLegendre *obj, FunctionArgs fargs) {
        auto& rets=fargs.rets;
        rets[0].x = obj->m_points;
        rets[1].x = Eigen::Map<const Eigen::ArrayXd>(&obj->m_edges[0], obj->m_edges.size());
      })
    .finalize()
    ;
}

GaussLegendreHist::GaussLegendreHist(const GaussLegendre *base)
  : m_base(base)
{
  transformation_("hist")
    .input("f")
    .output("hist")
    .types(TypesFunctions::ifSame, [](GaussLegendreHist *obj, TypesFunctionArgs fargs) {
        fargs.rets[0] = DataType().hist().bins(obj->m_base->m_orders.size()).edges(obj->m_base->m_edges);
      })
    .func([](GaussLegendreHist *obj, FunctionArgs fargs) {
        auto& ret=fargs.rets[0].x;
        ArrayXd prod = fargs.args[0].x*obj->m_base->m_weights;
        auto *data = prod.data();
        const auto &orders = obj->m_base->m_orders;
        for (size_t i = 0; i < orders.size(); ++i) {
          size_t n = orders[i];
          ret(i) = std::accumulate(data, data+n, 0.0);
          data += n;
        }
      })
    ;
}
