#include <gsl/gsl_integration.h>

#include <algorithm>

#include <Eigen/Dense>

#include "GaussLegendre2d.hh"

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
    .output("x", DataType().points().shape(m_xpoints.size()))
    .output("y", DataType().points().shape(m_ypoints.size()))
    .types(&GaussLegendre2d::pointsTypes)
    .func(&GaussLegendre2d::points);
  transformation_("hist")
    .input("f", DataType().points().any())
    .output("hist", DataType().hist().any())
    .types(&GaussLegendre2d::histTypes)
    .func(&GaussLegendre2d::hist);
}

void GaussLegendre2d::pointsTypes(Atypes /*args*/, Rtypes rets) {
  rets[0] = DataType().points().shape(m_xpoints.size());
  rets[1] = DataType().points().shape(m_ypoints.size());
}

void GaussLegendre2d::points(Args /*args*/, Rets rets) {
  rets[0].x = Eigen::Map<const Eigen::ArrayXd>(&m_xpoints[0], m_xpoints.size());
  rets[1].x = Eigen::Map<const Eigen::ArrayXd>(&m_ypoints[0], m_ypoints.size());
}

void GaussLegendre2d::histTypes(Atypes /*args*/, Rtypes rets) {
  rets[0] = DataType().hist().bins(m_xorders.size()).edges(m_xedges);
}

void GaussLegendre2d::hist(Args args, Rets rets) {
  size_t shape[2];
  shape[0] = m_xpoints.size();
  shape[1] = m_ypoints.size();
  Map<const ArrayXXd, Aligned> pts(args[0].x.data(), shape[0], shape[1]);
  auto &xw = m_xweights, &yw = m_yweights;
  ArrayXd prod = (pts.rowwise()*yw.transpose()).rowwise().sum()*xw;
  auto *data = prod.data();
  for (size_t i = 0; i < m_xorders.size(); ++i) {
    size_t n = m_xorders[i];
    rets[0].x(i) = std::accumulate(data, data+n, 0.0);
    data += n;
  }
}
