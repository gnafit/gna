#include "Rebin.hh"
#include <algorithm>
#include <functional>
#include <iterator>
#include <math.h>

using std::placeholders::_1;

Rebin::Rebin(size_t n, double* edges, int rounding) : m_new_edges(n), m_round_scale{pow(10, rounding)} {
  std::transform( edges, edges+n, m_new_edges.begin(), std::bind(&Rebin::round, this, _1) );

  transformation_("rebin")
    .input("histin")
    .output("histout")
    .types([](Rebin *obj, Atypes args, Rtypes rets){
           if(args[0].kind!=DataKind::Hist){
             throw std::runtime_error("Rebinner input should be a histogram");
           }
           rets[0]=DataType().hist().edges(obj->m_new_edges);
           })
    .func(&Rebin::calcSmear);
}

void Rebin::calcSmear(Args args, Rets rets) {
  if( !m_initialized ){
      calcMatrix( args[0].type );
  }
  rets[0].x = m_sparse_cache * args[0].vec;
}

void Rebin::calcMatrix(const DataType& type) {
  std::vector<double> edges(type.size()+1);
  std::transform( type.edges.begin(), type.edges.end(), edges.begin(), std::bind(&Rebin::round, this, _1) );

  m_sparse_cache.resize( m_new_edges.size()-1, type.size() );
  m_sparse_cache.setZero();

  auto edge_new = m_new_edges.begin();
  auto edge_old = std::lower_bound(edges.begin(), edges.end(), *edge_new);
  size_t iold=std::distance(edges.begin(), edge_old);
  for (size_t inew{0}; inew < m_new_edges.size(); ++inew) {
    while(*edge_old<*edge_new) {
      m_sparse_cache.insert(inew-1, iold) = 1.0;

      ++edge_old;
      ++iold;
      if(edge_old==edges.end()){
        throw std::runtime_error("Bin edges are not consistent (outer)");
      }
    }
    if(*edge_new!=*edge_old){
      throw std::runtime_error("Bin edges are not consistent (inner)");
    }
    ++edge_new;
  }
}

double Rebin::round(double num){
  return std::round( num*m_round_scale )/m_round_scale;
}
