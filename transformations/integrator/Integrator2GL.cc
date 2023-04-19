#include "Integrator2GL.hh"

#include <Eigen/Dense>
#include <map>
#include <iterator>

#include "GSLSamplerGL.hh"
#include "TypesFunctions.hh"

using namespace Eigen;
using namespace std;

Integrator2GL::Integrator2GL(size_t xbins, int xorders, size_t ybins, int  yorders) :
Integrator2GL(xbins, xorders, nullptr, ybins, yorders, nullptr)
{}

Integrator2GL::Integrator2GL(size_t xbins, int* xorders, size_t ybins, int* yorders) :
Integrator2GL(xbins, xorders, nullptr, ybins, yorders, nullptr)
{}

Integrator2GL::Integrator2GL(size_t xbins, int xorders, double* xedges, size_t ybins, int  yorders, double* yedges) :
Integrator2Base(xbins, xorders, xedges, ybins, yorders, yedges)
{
  init_sampler();
}

Integrator2GL::Integrator2GL(size_t xbins, int* xorders, double* xedges, size_t ybins, int* yorders, double* yedges) :
Integrator2Base(xbins, xorders, xedges, ybins, yorders, yedges)
{
  init_sampler();
}

void Integrator2GL::sample(FunctionArgs& fargs){
  auto& rets=fargs.rets;
  GSLSamplerGL sampler;
  auto& x=rets[0];
  auto& y=rets[1];
  sampler.fill_bins(m_xorders.size(), m_xorders.data(), m_xedges.data(), x.buffer, m_xweights.data());
  sampler.fill_bins(m_yorders.size(), m_yorders.data(), m_yedges.data(), y.buffer, m_yweights.data());

  m_weights = m_xweights.matrix() * m_yweights.matrix().transpose();

  rets[2].x = m_xedges.cast<double>();
  rets[3].x = m_yedges.cast<double>();

  rets[4].mat = x.vec.replicate(1, m_yweights.size());
  rets[5].mat = y.vec.transpose().replicate(m_xweights.size(), 1);
  rets[6].x = 0.0;
  rets[7].x = 0.0;
  auto xnpoints=m_xedges.size()-1;
  auto ynpoints=m_yedges.size()-1;
  rets[8].x = 0.5*(m_xedges.tail(xnpoints)+m_xedges.head(xnpoints));
  rets[9].x = 0.5*(m_yedges.tail(ynpoints)+m_yedges.head(ynpoints));
  rets.untaint();
  rets.freeze();
}
