#include "Integrator21GL.hh"

#include <Eigen/Dense>
#include <map>
#include <iterator>

#include "GSLSamplerGL.hh"
#include "TypesFunctions.hh"

using namespace Eigen;
using namespace std;

Integrator21GL::Integrator21GL(size_t xbins, int xorders, double* xedges, int yorder, double ymin, double ymax) :
Integrator21Base(xbins, xorders, xedges, yorder, ymin, ymax)
{
  init_sampler();
}

Integrator21GL::Integrator21GL(size_t xbins, int* xorders, double* xedges, int yorder, double ymin, double ymax) :
Integrator21Base(xbins, xorders, xedges, yorder, ymin, ymax)
{
  init_sampler();
}

void Integrator21GL::sample(FunctionArgs& fargs){
  auto* edge_a=m_xedges.data();
  auto* edge_b{next(edge_a)};
  auto& rets=fargs.rets;
  auto *abscissa(rets[0].buffer), *weight(m_xweights.data());
  rets[1].x = m_xedges.cast<double>();

  GSLSamplerGL sampler;
  for (size_t i = 0; i < m_xorders.size(); ++i) {
    size_t n = static_cast<size_t>(m_xorders[i]);
    sampler.fill(n, *edge_a, *edge_b, abscissa, weight);
    advance(abscissa, n);
    advance(weight, n);
    advance(edge_a, 1);
    advance(edge_b, 1);
  }

  abscissa=rets[1].buffer;
  weight=m_yweights.data();
  sampler.fill(m_yorder, m_ymin, m_ymax, abscissa, weight);
}

