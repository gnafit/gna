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
  sampler.fill_bins(m_xorders.size(), m_xorders.data(), m_xedges.data(), rets[0].buffer, m_xweights.data());
  sampler.fill(m_yorder, m_ymin, m_ymax, rets[1].buffer, m_yweights.data());
  rets[2].x = m_xedges.cast<double>();
}

