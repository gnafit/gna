#include "IntegratorRect.hh"

#include <Eigen/Dense>
#include <map>
#include <iterator>

#include "GSLSamplerGL.hh"
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

  auto trans=transformation_("points")
      .output("x")
      .output("xedges")
      .types(&IntegratorRect::check)
      .func(&IntegratorRect::compute_rect)
      ;

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

  if(m_edges.size()){
    trans.finalize();
  }
  else{
    trans.input("edges") //hist with edges
      .types(TypesFunctions::if1d<0>, TypesFunctions::ifHist<0>, TypesFunctions::binsToEdges<0,1>);
  }
}

void IntegratorRect::check(TypesFunctionArgs& fargs){
  auto& rets=fargs.rets;
  rets[0] = DataType().points().shape(m_weights.size());

  auto& args=fargs.args;
  if(args.size() && !m_edges.size()){
    auto& edges=fargs.args[0].edges;
    m_edges=Map<const ArrayXd>(edges.data(), edges.size());
  }
  rets[1]=DataType().points().shape(m_edges.size()).preallocated(m_edges.data());
}

void IntegratorRect::compute_rect(FunctionArgs& fargs){
  auto& rets=fargs.rets;
  rets[1].x = m_edges.cast<double>();
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
  for (size_t i = 0; i < m_orders.size(); ++i) {
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
