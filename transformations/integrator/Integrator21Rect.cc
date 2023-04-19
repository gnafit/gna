#include "Integrator21Rect.hh"

#include <Eigen/Dense>
#include <iterator>

#include "SamplerRect.hh"
#include "TypesFunctions.hh"

using namespace Eigen;
using namespace std;

Integrator21Rect::Integrator21Rect(size_t xbins, int xorders, double* xedges, int yorder, double ymin, double ymax, const std::string& mode) :
Integrator21Base(xbins, xorders, xedges, yorder, ymin, ymax),
m_mode(mode)
{
    m_rect_offset = SamplerRect::offset(mode);
    init_sampler();
}

Integrator21Rect::Integrator21Rect(size_t xbins, int* xorders, double* xedges, int yorder, double ymin, double ymax, const std::string& mode) :
Integrator21Base(xbins, xorders, xedges, yorder, ymin, ymax),
m_mode(mode)
{
    m_rect_offset = SamplerRect::offset(mode);
    init_sampler();
}

void Integrator21Rect::sample(FunctionArgs& fargs){
    auto& rets=fargs.rets;
    auto& x=rets[0];
    auto& y=rets[1];
    SamplerRect::fill_bins(m_rect_offset, m_xorders.size(), m_xorders.data(), m_xedges.data(), x.buffer, m_xweights.data());
    SamplerRect::fill(m_rect_offset, m_yorder, m_ymin, m_ymax, y.buffer, m_yweights.data());

    m_weights = m_xweights.matrix() * m_yweights.matrix().transpose();

    rets[2].x = m_xedges.cast<double>();

    auto npoints=m_xedges.size()-1;
    rets[3].x = 0.5*(m_xedges.tail(npoints)+m_xedges.head(npoints));
    rets[4].mat = x.vec.replicate(1, m_yweights.size());
    rets[5].mat = y.vec.transpose().replicate(m_xweights.size(), 1);
    rets[6].x = 0.0;

    rets.untaint();
    rets.freeze();
}
