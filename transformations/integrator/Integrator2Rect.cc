#include "Integrator2Rect.hh"

#include <Eigen/Dense>
#include <iterator>

#include "SamplerRect.hh"
#include "TypesFunctions.hh"

using namespace Eigen;
using namespace std;

Integrator2Rect::Integrator2Rect(size_t xbins, int xorders, size_t ybins, int  yorders, const std::string& mode) :
Integrator2Rect(xbins, xorders, nullptr, ybins, yorders, nullptr, mode)
{}

Integrator2Rect::Integrator2Rect(size_t xbins, int* xorders, size_t ybins, int* yorders, const std::string& mode) :
Integrator2Rect(xbins, xorders, nullptr, ybins, yorders, nullptr, mode)
{}

Integrator2Rect::Integrator2Rect(size_t xbins, int xorders, double* xedges, size_t ybins, int  yorders, double* yedges, const std::string& mode) :
Integrator2Base(xbins, xorders, xedges, ybins, yorders, yedges),
m_mode(mode)
{
    m_rect_offset = SamplerRect::offset(mode);
    init_sampler();
}

Integrator2Rect::Integrator2Rect(size_t xbins, int* xorders, double* xedges, size_t ybins, int* yorders, double* yedges, const std::string& mode) :
Integrator2Base(xbins, xorders, xedges, ybins, yorders, yedges),
m_mode(mode)
{
    m_rect_offset = SamplerRect::offset(mode);
    init_sampler();
}

void Integrator2Rect::sample(FunctionArgs& fargs){
    auto& rets=fargs.rets;
    auto& x=rets[0];
    auto& y=rets[1];
    SamplerRect::fill_bins(m_rect_offset, m_xorders.size(), m_xorders.data(), m_xedges.data(), x.buffer, m_xweights.data());
    SamplerRect::fill_bins(m_rect_offset, m_yorders.size(), m_yorders.data(), m_yedges.data(), y.buffer, m_yweights.data());

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
