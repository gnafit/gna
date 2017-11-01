#include "Rebin.hh"
#include <algorithm>
#include <functional>
#include <math.h>

using std::placeholders::_1;

Rebin::Rebin(size_t n, double* edges, int rounding) : m_new_edges(n), m_round_scale{pow(10, rounding)} {
  std::transform( edges, edges+n, m_new_edges.begin(), std::bind(&Rebin::round, this, _1) );

  transformation_(this, "rebin")
    .input("histin")
    .output("histout")
    .types(&Rebin::calcMatrix)
    .func(&Rebin::calcSmear);
}

void Rebin::calcSmear(Args args, Rets rets) {
  rets[0].x = m_sparse_cache * args[0].vec;
}

void Rebin::calcMatrix(Atypes args, Rtypes rets) {
  if(args[0].kind!=DataKind::Hist){
    throw std::runtime_error("Input should be a histogram");
  }
  rets[0]=DataType().hist().bins(m_new_edges.size()-1).edges(m_new_edges);

  std::vector<double> edges(args[0].size());
  std::transform( args[0].edges.begin(), args[0].edges.end(), edges.begin(), std::bind(&Rebin::round, this, _1) );

  m_sparse_cache.resize( rets[0].size(), args[0].size() );
  m_sparse_cache.setZero();

  auto edge_new = m_new_edges.begin();
  auto edge_old = edges.begin();
  size_t iold{0};
  for (size_t inew{0}; inew < m_new_edges.size(); ++inew) {
    printf("old %lu %.3f new %lu %.3f, diff %g\n", iold, *edge_old, inew, *edge_new, *edge_new-*edge_old);
    while(*edge_old<*edge_new) {
      m_sparse_cache.insert(inew, iold) = 1.0;
      printf("  old %lu %.3f new %lu %.3f, diff %g\n", iold, *edge_old, inew, *edge_new, *edge_new-*edge_old);

      ++edge_old;
      ++iold;
      //if(edge_old==edges.end()){
        //throw std::runtime_error("Bin edges are not consistent (outer)");
      //}
    }
    printf("old %lu %.3f new %lu %.3f, diff %g\n\n", iold, *edge_old, inew, *edge_new, *edge_new-*edge_old);
    if(*edge_new!=*edge_old){
      throw std::runtime_error("Bin edges are not consistent (inner)");
    }
    ++edge_new;
  }
}

double Rebin::round(double num){
  return std::round( num*m_round_scale )/m_round_scale;
}
