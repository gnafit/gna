#include "Integrator2Rect.hh"

#include <Eigen/Dense>
#include <iterator>

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
Integrator2Base(xbins, xorders, xedges, ybins, yorders, yedges)
{
  init(mode);
}

Integrator2Rect::Integrator2Rect(size_t xbins, int* xorders, double* xedges, size_t ybins, int* yorders, double* yedges, const std::string& mode) :
Integrator2Base(xbins, xorders, xedges, ybins, yorders, yedges)
{
  init(mode);
}

void Integrator2Rect::init(const std::string& mode) {
  m_mode = mode;

  if(m_mode=="left"){
    m_rect_offset=-1;
  }
  else if(m_mode=="center"){
    m_rect_offset=0;
  }
  else if(m_mode=="right"){
    m_rect_offset=1;
  }
  else{
    throw std::runtime_error("invalid rectangular integration mode");
  }

  init_sampler();
}

void Integrator2Rect::sample(FunctionArgs& fargs){
  auto& rets=fargs.rets;

  auto& x=rets[0].x;
  auto& y=rets[1].x;


  auto xnbins=m_xedges.size()-1;
  auto ynbins=m_yedges.size()-1;
  auto& xbinwidths=m_xedges.tail(xnbins) - m_xedges.head(xnbins);
  auto& ybinwidths=m_yedges.tail(ynbins) - m_yedges.head(ynbins);
  ArrayXd xsamplewidths=xbinwidths/m_xorders.cast<double>();
  ArrayXd ysamplewidths=ybinwidths/m_yorders.cast<double>();

  ArrayXd xlow, ylow, xhigh, yhigh;
  switch(m_rect_offset){
    case -1: {
      xlow=m_xedges.head(xnbins);
      ylow=m_yedges.head(ynbins);
      xhigh=m_xedges.tail(xnbins)-xsamplewidths;
      yhigh=m_yedges.tail(ynbins)-ysamplewidths;
      break;
      }
    case 0: {
      ArrayXd xoffsetwidth=xsamplewidths*0.5;
      ArrayXd yoffsetwidth=ysamplewidths*0.5;
      xlow=m_xedges.head(xnbins)+xoffsetwidth;
      ylow=m_yedges.head(ynbins)+yoffsetwidth;
      xhigh=m_xedges.tail(xnbins)-xoffsetwidth;
      yhigh=m_yedges.tail(ynbins)-yoffsetwidth;
      break;
      }
    case 1: {
      xlow=m_xedges.head(xnbins)+xsamplewidths;
      ylow=m_yedges.head(ynbins)+ysamplewidths;
      xhigh=m_xedges.tail(xnbins);
      yhigh=m_yedges.tail(ynbins);
      break;
      }
  }

  size_t offset=0;
  for (size_t i = 0; i < static_cast<size_t>(m_xorders.size()); ++i) {
    auto n=m_xorders[i];
    if(n>1){
      x.segment(offset, n)=ArrayXd::LinSpaced(n, xlow[i], xhigh[i]);
      m_xweights.segment(offset, n)=xsamplewidths[i];
    }
    else{
      x[i]=xlow[i];
      m_xweights[i]=xbinwidths[i];
    }
    offset+=n;
  }

  offset=0;
  for (size_t i = 0; i < static_cast<size_t>(m_yorders.size()); ++i) {
    auto n=m_yorders[i];
    if(n>1){
      y.segment(offset, n)=ArrayXd::LinSpaced(n, ylow[i], yhigh[i]);
      m_yweights.segment(offset, n)=ysamplewidths[i];
    }
    else{
      y[i]=ylow[i];
      m_yweights[i]=ybinwidths[i];
    }
    offset+=n;
  }

  m_weights = m_xweights.matrix() * m_yweights.matrix().transpose();

  rets[2].x = m_xedges.cast<double>();
  rets[3].x = m_yedges.cast<double>();

  rets[4].mat = x.replicate(1, m_yweights.size());
  rets[5].mat = y.transpose().replicate(m_xweights.size(), 1);

  rets[6].x = 0.0;
  rets[7].x = 0.0;

  rets[8].x = 0.5*(m_xedges.tail(xnbins)+m_xedges.head(xnbins));
  rets[9].x = 0.5*(m_yedges.tail(ynbins)+m_yedges.head(ynbins));

  rets.untaint();
  rets.freeze();
}
