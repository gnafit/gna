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
  auto& rets=fargs.rets;
  GSLSamplerGL sampler;
  auto& x=rets[0];
  auto& y=rets[1];
  sampler.fill_bins(m_xorders.size(), m_xorders.data(), m_xedges.data(), x.buffer, m_xweights.data());
  sampler.fill(m_yorder, m_ymin, m_ymax, y.buffer, m_yweights.data());

  m_weights = m_xweights.matrix() * m_yweights.matrix().transpose();

  rets[2].x = m_xedges.cast<double>();

  rets[3].mat = x.vec.replicate(1, m_yweights.size());
  rets[4].mat = y.vec.transpose().replicate(m_xweights.size(), 1);
  rets[5].x = 0.0;

  rets.untaint();
  rets.freeze();
}

