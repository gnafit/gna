#include "IntegratorRect.hh"

#include <Eigen/Dense>
#include <iterator>

#include "TypesFunctions.hh"

using namespace Eigen;
using namespace std;

IntegratorRect::IntegratorRect(size_t bins, int orders, double* edges, const std::string& mode) : IntegratorBase(bins, orders, edges)
{
  init(mode);
}

IntegratorRect::IntegratorRect(size_t bins, int* orders, double* edges, const std::string& mode) : IntegratorBase(bins, orders, edges)
{
  init(mode);
}

void IntegratorRect::init(const std::string& mode) {
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

void IntegratorRect::sample(FunctionArgs& fargs){
  auto& rets=fargs.rets;
  rets[1].x = m_edges.cast<double>();
  auto npoints=m_edges.size()-1;
  rets[3].x = 0.5*(m_edges.tail(npoints)+m_edges.head(npoints));

  auto& abscissa=rets[0].x;

  auto nbins=m_edges.size()-1;
  auto& binwidths=m_edges.tail(nbins) - m_edges.head(nbins);
  ArrayXd samplewidths=binwidths/m_orders.cast<double>();

  ArrayXd low, high;
  switch(m_rect_offset){
    case -1: {
      low=m_edges.head(nbins);
      high=m_edges.tail(nbins)-samplewidths;
      break;
      }
    case 0: {
      ArrayXd offsetwidth=samplewidths*0.5;
      low=m_edges.head(nbins)+offsetwidth;
      high=m_edges.tail(nbins)-offsetwidth;
      break;
      }
    case 1: {
      samplewidths=binwidths/m_orders.cast<double>();
      low=m_edges.head(nbins)+samplewidths;
      high=m_edges.tail(nbins);
      break;
      }
  }

  size_t offset=0;
  for (size_t i = 0; i < static_cast<size_t>(m_orders.size()); ++i) {
    auto n=m_orders[i];
    if(n>1){
      abscissa.segment(offset, n)=ArrayXd::LinSpaced(n, low[i], high[i]);
      m_weights.segment(offset, n)=samplewidths[i];
    }
    else{
      abscissa[i]=low[i];
      m_weights[i]=binwidths[i];
    }
    offset+=n;
  }
}
