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

void Integrator2Rect::calcWeights(int rect_offset, size_t nbins, int* order, double* edges, double* abscissa, double* weight){

  auto* edge_a=edges;
  auto* edge_b=next(edges);
  for (size_t i=0; i < nbins; ++i) {

    auto binwidth=*edge_b - *edge_a;
    auto samplewidth=binwidth / *order;

    double low=0, high=0;
    switch(rect_offset){
      case -1: {
        low=*edge_a;
        high=*edge_b-samplewidth;
        break;
        }
      case 0: {
        double offsetwidth=samplewidth*0.5;
        low=*edge_a+offsetwidth;
        high=*edge_b-offsetwidth;
        break;
        }
      case 1: {
        low=*edge_a+samplewidth;
        high=*edge_b;
        break;
        }
    }

  

    if(*order>1){
      double step = (high - low)/(*order-1);
      for (int i = 0; i < *order; ++i) {   
        *abscissa=low + i*step;
        *weight=samplewidth;
        advance(abscissa,1);
        advance(weight,1);
      }
    }
    else{
      *abscissa=low;
      *weight=binwidth;
    }

    advance(order, 1);
    advance(edge_a, 1);
    advance(edge_b, 1);

  }
}


void Integrator2Rect::sample(FunctionArgs& fargs){
  auto& rets=fargs.rets;

  auto& x=rets[0];
  auto& y=rets[1];

  auto xnbins=m_xedges.size()-1;
  auto ynbins=m_yedges.size()-1;

  calcWeights(m_rect_offset, m_xorders.size(), m_xorders.data(), m_xedges.data(), x.buffer, m_xweights.data());
  calcWeights(m_rect_offset, m_yorders.size(), m_yorders.data(), m_yedges.data(), y.buffer, m_yweights.data());

  m_weights = m_xweights.matrix() * m_yweights.matrix().transpose();

  rets[2].x = m_xedges.cast<double>();
  rets[3].x = m_yedges.cast<double>();

  rets[4].mat = x.vec.replicate(1, m_yweights.size());
  rets[5].mat = y.vec.transpose().replicate(m_xweights.size(), 1);

  rets[6].x = 0.0;
  rets[7].x = 0.0;

  rets[8].x = 0.5*(m_xedges.tail(xnbins)+m_xedges.head(xnbins));
  rets[9].x = 0.5*(m_yedges.tail(ynbins)+m_yedges.head(ynbins));

  rets.untaint();
  rets.freeze();
}

